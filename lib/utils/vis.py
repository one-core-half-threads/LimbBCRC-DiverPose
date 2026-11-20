# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import torch

from core.inference import get_max_preds


NUM_KPTS = 16

# 蛙人 MPII16 顺序：
# 0 rank, 1 rkne, 2 rhip, 3 lhip, 4 lkne, 5 lank,
# 6 root, 7 spine, 8 thorax, 9 head, 10 rwri, 11 relb, 12 rsho, 13 lsho, 14 lelb, 15 lwri
SKELETON = [
    (0, 1), (1, 2), (2, 6),       # 右腿
    (5, 4), (4, 3), (3, 6),       # 左腿
    (6, 7), (7, 8), (8, 9),       # 躯干 + 头
    (8, 12), (12, 11), (11, 10),  # 右臂
    (8, 13), (13, 14), (14, 15),  # 左臂
]

# OpenCV BGR 颜色，和单独推理脚本中的 draw_pose 保持一致。
SKELETON_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (255, 0, 255), (255, 0, 255), (255, 0, 255),
]


def _denormalize_batch_image(batch_image):
    """ImageNet mean/std 反归一化，仅用于 debug 可视化。"""
    image = batch_image.detach().float().cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype).view(1, 3, 1, 1)
    image = image * std + mean
    return image.clamp(0.0, 1.0)


def _tensor_image_to_uint8(image_tensor, color_rgb=True):
    """单张 CHW tensor -> OpenCV 可写入/绘制的 BGR uint8 图像。"""
    image = image_tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    image = image.copy()
    if color_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _valid_joint_xy(joint):
    x, y = float(joint[0]), float(joint[1])
    if x < 0 or y < 0 or (not np.isfinite(x)) or (not np.isfinite(y)):
        return None
    return int(round(x)), int(round(y))


def _joint_visible(joints_vis, joint_id, draw_invisible=False):
    if joint_id >= len(joints_vis):
        return bool(draw_invisible)
    return bool(float(joints_vis[joint_id][0]) > 0) or bool(draw_invisible)


def _draw_pose_on_image(img, keypoints, joints_vis, draw_invisible=False):
    """按照单独推理脚本的方式绘制骨骼线、关节点和编号。"""
    if keypoints is None:
        return img

    num_joints = min(len(keypoints), NUM_KPTS)

    for limb_idx, (a, b) in enumerate(SKELETON):
        if a >= num_joints or b >= num_joints:
            continue
        if not (_joint_visible(joints_vis, a, draw_invisible) and
                _joint_visible(joints_vis, b, draw_invisible)):
            continue
        pa = _valid_joint_xy(keypoints[a])
        pb = _valid_joint_xy(keypoints[b])
        if pa is None or pb is None:
            continue
        color = SKELETON_COLORS[limb_idx % len(SKELETON_COLORS)]
        cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)

    for jid in range(num_joints):
        if not _joint_visible(joints_vis, jid, draw_invisible):
            continue
        p = _valid_joint_xy(keypoints[jid])
        if p is None:
            continue
        x, y = p
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(jid), (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2,
                                 draw_invisible=False, color_rgb=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 2/3]
    batch_joints_vis: [batch_size, num_joints, 1]

    Debug 可视化专用：先反归一化，再根据 DATASET.COLOR_RGB 决定是否 RGB->BGR，
    避免 cv2.imwrite 保存验证集图片时出现水下图像偏黄/偏色。
    '''
    batch_image_vis = _denormalize_batch_image(batch_image)

    nmaps = batch_image_vis.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    image_h = int(batch_image_vis.size(2))
    image_w = int(batch_image_vis.size(3))
    cell_h = int(image_h + padding)
    cell_w = int(image_w + padding)

    grid = torchvision.utils.make_grid(
        batch_image_vis, nrow=nrow, padding=padding, normalize=False
    )
    ndarr = _tensor_image_to_uint8(grid, color_rgb=color_rgb)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]
            joints_np = joints.detach().cpu().numpy() if torch.is_tensor(joints) else np.asarray(joints)
            joints_vis_np = joints_vis.detach().cpu().numpy() if torch.is_tensor(joints_vis) else np.asarray(joints_vis)

            offset_x = x * cell_w + padding
            offset_y = y * cell_h + padding
            joints_draw = joints_np.copy()
            joints_draw[:, 0] = joints_draw[:, 0] + offset_x
            joints_draw[:, 1] = joints_draw[:, 1] + offset_y
            _draw_pose_on_image(ndarr, joints_draw, joints_vis_np, draw_invisible=draw_invisible)
            k = k + 1

    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True, color_rgb=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = _denormalize_batch_image(batch_image)
    else:
        batch_image = batch_image.detach().float().cpu().clamp(0.0, 1.0)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = _tensor_image_to_uint8(batch_image[i], color_rgb=color_rgb)
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_masks(batch_image, batch_masks, file_name,
                        normalize=True, color_rgb=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = _denormalize_batch_image(batch_image)
    else:
        batch_image = batch_image.detach().float().cpu().clamp(0.0, 1.0)

    batch_size = batch_masks.size(0)
    num_joints = batch_masks.size(1)
    heatmap_height = batch_masks.size(2)
    heatmap_width = batch_masks.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)


    for i in range(batch_size):
        image = _tensor_image_to_uint8(batch_image[i], color_rgb=color_rgb)
        heatmaps = batch_masks[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):

            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, mask_pred=None, draw_invisible=False):
    if not config.DEBUG.DEBUG:
        return

    color_rgb = bool(config.DATASET.COLOR_RGB)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix),
            draw_invisible=draw_invisible,
            color_rgb=color_rgb
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix),
            draw_invisible=draw_invisible,
            color_rgb=color_rgb
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix),
            color_rgb=color_rgb
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix),
            color_rgb=color_rgb
        )



    if config.DEBUG.SAVE_BATCH_IMAGES_GT and (mask_pred is not None):
        save_batch_masks(
            input, meta['mask_target'], '{}_mask_gt.jpg'.format(prefix),
            color_rgb=color_rgb
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED and (mask_pred is not None):
        save_batch_masks(
            input, mask_pred, '{}_mask_pred.jpg'.format(prefix),
            color_rgb=color_rgb
        )