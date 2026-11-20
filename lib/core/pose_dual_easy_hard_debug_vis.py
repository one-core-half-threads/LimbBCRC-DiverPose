# -*- coding: utf-8 -*-
"""
PoseDual / HeteroLimb 训练过程可视化函数。v6：
1. 使用 easy_x1/easy_x 作为 easy teacher 预测背景图。
2. Net1 easy 每个关节点的置信度显示在右侧独立信息面板。
3. 火柴人绘制更细，尽量减少对原始人体图像的遮挡。
4. 对 pose_hetero_limb 额外显示肢体热度图与对应图像的叠加结果。

用途：
放到 function.py 中，或者另存为一个 py 文件后在 function.py 中导入。
建议插入位置：

    if config.MODEL.NAME in ['pose_dual', 'pose_cons']:
        output = model(input, aug_couple=aug_couple)
    else:
        output = model(input)

    debug_visualize_pose_dual_easy_hard(
        output,
        save_dir=os.path.join(output_dir, 'easy_hard_debug'),
        max_images=2,
        save_every=200,
    )

显示内容：
1. Net1 的 easy augmentation 伪标签预测结果：unsup_ht_trans1
2. Net2 的 hard path 0 预测结果：cons_ht2_list[0]，通常对应 JOCO
3. Net2 的 hard path 1 预测结果：cons_ht2_list[1]，通常对应 JCCM
4. Net1 easy 每个关节点的置信度：显示在右侧独立信息面板

说明：
- v5 需要配合修改后的 pose_dual.py 使用。pose_dual.py 会在 out_dic 中返回：
  easy_x1: Net1 的 easy augmentation 输入图像，且已经与 unsup_ht_trans1 对齐；
  easy_x2: Net2 的 easy augmentation 输入图像，且已经与 unsup_ht_trans2 对齐。
- 本函数只 detach 结果做可视化，不参与反向传播。
"""

import os
import re
import time

import cv2
import numpy as np
import torch


NUM_KPTS = 16

# 你的蛙人 MPII16 顺序：
# 0 rank, 1 rkne, 2 rhip, 3 lhip, 4 lkne, 5 lank,
# 6 root, 7 spine, 8 thorax, 9 head,
# 10 rwri, 11 relb, 12 rsho, 13 lsho, 14 lelb, 15 lwri
SKELETON = [
    (0, 1), (1, 2), (2, 6),       # right leg
    (5, 4), (4, 3), (3, 6),       # left leg
    (6, 7), (7, 8), (8, 9),       # torso + head
    (8, 12), (12, 11), (11, 10),  # right arm
    (8, 13), (13, 14), (14, 15),  # left arm
]

# BGR colors for OpenCV
LIMB_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0),       # right leg
    (255, 0, 0), (255, 0, 0), (255, 0, 0),       # left leg
    (0, 255, 255), (0, 255, 255), (0, 255, 255), # torso
    (0, 0, 255), (0, 0, 255), (0, 0, 255),       # right arm
    (255, 0, 255), (255, 0, 255), (255, 0, 255), # left arm
]

JOINT_NAMES = [
    "rank", "rkne", "rhip", "lhip", "lkne", "lank",
    "root", "spine", "thorax", "head",
    "rwri", "relb", "rsho", "lsho", "lelb", "lwri"
]

# 肢体热图通道顺序与 utils/limb_heatmap.py 中 MPII_PARENT_IDS 生成的 limb_pairs 一致：
# 00 rank-rkne, 01 rkne-rhip, 02 rhip-root,
# 03 lhip-root, 04 lkne-lhip, 05 lank-lkne,
# 06 spine-root, 07 thorax-spine, 08 head-thorax,
# 09 rwri-relb, 10 relb-rsho, 11 rsho-thorax,
# 12 lsho-thorax, 13 lelb-lsho, 14 lwri-lelb.
RIGHT_LIMB_IDS = {0, 1, 2, 9, 10, 11}
LEFT_LIMB_IDS = {3, 4, 5, 12, 13, 14}
CENTER_LIMB_IDS = {6, 7, 8}

# OpenCV 使用 BGR。用户要求：左侧肢体蓝色十字，右侧肢体绿色十字。
LIMB_CENTER_COLOR_LEFT = (255, 0, 0)      # blue, BGR
LIMB_CENTER_COLOR_RIGHT = (0, 255, 0)     # green, BGR
LIMB_CENTER_COLOR_CENTER = (0, 255, 255)  # yellow, BGR, for torso/head center-line limbs


def _is_main_process_from_env():
    """
    避免 DDP 多进程同时保存重复图片。
    非分布式训练时 LOCAL_RANK 通常不存在，默认认为是主进程。
    """
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    try:
        return int(rank) == 0
    except Exception:
        return True


def _tensor_img_to_bgr(img_tensor, tensor_is_rgb=True):
    """
    反归一化训练输入 tensor，转成 OpenCV BGR uint8 图像。
    """
    if img_tensor is None:
        return None

    img = img_tensor.detach().float().cpu()

    if img.ndim != 3 or img.shape[0] != 3:
        return None

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)

    img = img * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    if tensor_is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def _heatmap_to_points_and_scores(heatmap, img_w, img_h):
    """
    heatmap -> 关键点坐标与置信度。
    """
    if not torch.is_tensor(heatmap):
        raise TypeError("heatmap must be torch.Tensor")

    hm = heatmap.detach().float().cpu()
    if hm.ndim != 4:
        raise ValueError("heatmap shape should be [B,J,H,W], got {}".format(tuple(hm.shape)))

    b, j, h, w = hm.shape
    flat = hm.reshape(b, j, -1)
    scores, idx = torch.max(flat, dim=2)

    xs = (idx % w).float()
    ys = torch.div(idx, w, rounding_mode='floor').float()

    xs = xs * (float(img_w) / float(w))
    ys = ys * (float(img_h) / float(h))

    points = torch.stack([xs, ys], dim=2).numpy()
    scores = scores.numpy()

    return points, scores


def _draw_pose(img, keypoints, scores=None, title=None,
               conf_thr_draw=0.0,
               line_thickness=1,
               joint_radius=2,
               text_scale=0.35,
               joint_text=True):
    """
    在图像上画骨骼和关节点编号。
    - 不在图中显示置信度
    - 线和点更细，尽量减少遮挡
    """
    canvas = img.copy()
    h, w = canvas.shape[:2]

    if title:
        cv2.rectangle(canvas, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(canvas, title, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if scores is None:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    for limb_idx, (a, b) in enumerate(SKELETON):
        if a >= keypoints.shape[0] or b >= keypoints.shape[0]:
            continue

        if scores[a] < conf_thr_draw or scores[b] < conf_thr_draw:
            continue

        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        if not (np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb)):
            continue

        color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
        cv2.line(canvas, (int(round(xa)), int(round(ya))),
                 (int(round(xb)), int(round(yb))), color, line_thickness, cv2.LINE_AA)

    for jid, (x, y) in enumerate(keypoints):
        if jid >= len(scores) or scores[jid] < conf_thr_draw:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        x_i, y_i = int(round(x)), int(round(y))
        cv2.circle(canvas, (x_i, y_i), joint_radius, (0, 0, 255), -1, cv2.LINE_AA)

        if joint_text:
            cv2.putText(canvas, str(jid), (x_i + 3, y_i - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def _normalize_heatmap_2d(hm2d):
    """
    将单张 2D heatmap 归一化到 [0, 1]，用于可视化。
    兼容网络输出中存在少量负值的情况。
    """
    hm = np.asarray(hm2d, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)

    # 优先显示正响应；如果整体没有正响应，再退化为 min-max 可视化。
    hm_pos = np.maximum(hm, 0.0)
    max_pos = float(hm_pos.max()) if hm_pos.size > 0 else 0.0

    if max_pos > 1e-8:
        vals = hm_pos[hm_pos > 0]
        if vals.size > 0:
            denom = float(np.percentile(vals, 99.0))
        else:
            denom = max_pos
        denom = max(denom, 1e-8)
        norm = np.clip(hm_pos / denom, 0.0, 1.0)
        return norm

    mn = float(hm.min()) if hm.size > 0 else 0.0
    mx = float(hm.max()) if hm.size > 0 else 0.0
    if mx - mn <= 1e-8:
        return np.zeros_like(hm, dtype=np.float32)

    norm = (hm - mn) / (mx - mn)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _limb_heatmap_to_overlay(img, limb_heatmap, title=None, max_alpha=0.45):
    """
    将肢体热度图叠加到对应图像上。

    limb_heatmap 支持：
        [C, H, W]：多通道肢体热图，按通道 max 聚合；
        [H, W]：单通道热图。
    """
    if img is None or limb_heatmap is None:
        return None

    hm = limb_heatmap.detach().float().cpu() if torch.is_tensor(limb_heatmap) else limb_heatmap
    if torch.is_tensor(hm):
        hm = hm.numpy()
    hm = np.asarray(hm, dtype=np.float32)

    if hm.ndim == 3:
        # 多个肢体通道合成一张总响应图，便于快速观察与图像是否对齐。
        hm2d = np.max(hm, axis=0)
    elif hm.ndim == 2:
        hm2d = hm
    else:
        canvas = img.copy()
        if title:
            h, w = canvas.shape[:2]
            cv2.rectangle(canvas, (0, 0), (w, 24), (0, 0, 0), -1)
            cv2.putText(canvas, title + " | invalid limb heatmap", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    norm = _normalize_heatmap_2d(hm2d)
    h, w = img.shape[:2]
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_u8 = cv2.resize(norm_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    heat_color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    alpha = (norm_u8.astype(np.float32) / 255.0) * float(max_alpha)
    alpha = alpha[..., None]

    canvas = img.astype(np.float32) * (1.0 - alpha) + heat_color.astype(np.float32) * alpha
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # 在每个肢体热图通道上估计肢体段中心点，并画到叠加图上。
    # 左侧肢体：蓝色十字；右侧肢体：绿色十字；中轴肢体：黄色十字。
    if hm.ndim == 3:
        canvas = _draw_limb_heatmap_centers(canvas, hm, rel_thresh=0.35, min_area=5)

    if title:
        cv2.rectangle(canvas, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(canvas, title, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas



def _limb_center_color(limb_id):
    """
    根据 limb heatmap 通道编号返回中心十字颜色。
    左侧肢体：蓝色；右侧肢体：绿色；躯干/头部中轴肢体：黄色。
    """
    if int(limb_id) in LEFT_LIMB_IDS:
        return LIMB_CENTER_COLOR_LEFT
    if int(limb_id) in RIGHT_LIMB_IDS:
        return LIMB_CENTER_COLOR_RIGHT
    return LIMB_CENTER_COLOR_CENTER


def _estimate_limb_center_from_heatmap_pca(
        hm2d,
        rel_thresh=0.35,
        abs_thresh=1e-6,
        min_area=5):
    """
    只根据单个肢体热度图估计该肢体段的几何中心点。

    方法：
        正响应区域阈值筛选
        -> 最大连通域
        -> 对最大连通域像素做加权 PCA
        -> 沿主轴取 2% / 98% 分位端点
        -> 两端点中点作为肢体中心点

    Args:
        hm2d: np.ndarray, shape [H, W]，单个 limb heatmap。
        rel_thresh: 相对阈值，保留 max_val * rel_thresh 以上区域。
        abs_thresh: 最小响应阈值，防止全零热图。
        min_area: 最大连通域最小面积，过小则认为不可靠。

    Returns:
        center: (cx, cy)，热图坐标系下的中心点；失败时为 None。
        info: dict，调试信息。
    """
    hm = np.asarray(hm2d, dtype=np.float32)
    if hm.ndim != 2:
        return None, {"valid": False, "reason": "not 2d"}

    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)

    # 只使用正响应估计中心，避免负值背景参与。
    hm_pos = np.maximum(hm, 0.0)
    max_val = float(hm_pos.max()) if hm_pos.size > 0 else 0.0
    if max_val <= abs_thresh:
        return None, {"valid": False, "reason": "low max response", "max_val": max_val}

    thresh = max(max_val * float(rel_thresh), float(abs_thresh))
    binary = (hm_pos >= thresh).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None, {"valid": False, "reason": "no component", "max_val": max_val}

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])

    if largest_area < int(min_area):
        return None, {
            "valid": False,
            "reason": "component too small",
            "max_val": max_val,
            "area": largest_area,
        }

    mask = (labels == largest_label)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, {"valid": False, "reason": "empty component", "max_val": max_val}

    weights = hm_pos[ys, xs].astype(np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= abs_thresh:
        return None, {
            "valid": False,
            "reason": "low component weight",
            "max_val": max_val,
            "area": largest_area,
        }

    # 加权均值作为 PCA 原点。
    x_mean = float((xs.astype(np.float32) * weights).sum() / weight_sum)
    y_mean = float((ys.astype(np.float32) * weights).sum() / weight_sum)

    if xs.size < 2:
        return (x_mean, y_mean), {
            "valid": True,
            "method": "weighted_centroid_single_pixel",
            "max_val": max_val,
            "area": largest_area,
        }

    pts = np.stack(
        [xs.astype(np.float32) - x_mean, ys.astype(np.float32) - y_mean],
        axis=1,
    )

    w_norm = weights / max(weight_sum, abs_thresh)
    cov = (pts * w_norm[:, None]).T @ pts

    if not np.all(np.isfinite(cov)):
        return (x_mean, y_mean), {
            "valid": True,
            "method": "weighted_centroid_bad_cov",
            "max_val": max_val,
            "area": largest_area,
        }

    eigvals, eigvecs = np.linalg.eigh(cov)
    main_id = int(np.argmax(eigvals))
    main_axis = eigvecs[:, main_id].astype(np.float32)

    axis_norm = float(np.linalg.norm(main_axis))
    if axis_norm <= 1e-8 or float(eigvals[main_id]) <= 1e-8:
        return (x_mean, y_mean), {
            "valid": True,
            "method": "weighted_centroid_degenerate_pca",
            "max_val": max_val,
            "area": largest_area,
        }
    main_axis = main_axis / axis_norm

    # 沿主轴投影，用分位数端点降低孤立噪声影响。
    proj = pts @ main_axis
    p_min = float(np.percentile(proj, 2.0))
    p_max = float(np.percentile(proj, 98.0))

    endpoint1 = np.array([x_mean, y_mean], dtype=np.float32) + p_min * main_axis
    endpoint2 = np.array([x_mean, y_mean], dtype=np.float32) + p_max * main_axis
    center = (endpoint1 + endpoint2) * 0.5

    return (float(center[0]), float(center[1])), {
        "valid": True,
        "method": "largest_component_pca_midpoint",
        "max_val": max_val,
        "area": largest_area,
        "threshold": float(thresh),
        "endpoint1": (float(endpoint1[0]), float(endpoint1[1])),
        "endpoint2": (float(endpoint2[0]), float(endpoint2[1])),
    }


def _draw_cross_marker(canvas, x, y, color, size=7, thickness=2):
    """
    在图像上画十字。先画黑色外轮廓，再画指定颜色，保证在热度图上可见。
    """
    if canvas is None:
        return
    if not (np.isfinite(x) and np.isfinite(y)):
        return

    h, w = canvas.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return

    size = int(size)
    thickness = int(thickness)
    outline = (0, 0, 0)

    cv2.line(canvas, (xi - size, yi), (xi + size, yi), outline, thickness + 2, cv2.LINE_AA)
    cv2.line(canvas, (xi, yi - size), (xi, yi + size), outline, thickness + 2, cv2.LINE_AA)
    cv2.line(canvas, (xi - size, yi), (xi + size, yi), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (xi, yi - size), (xi, yi + size), color, thickness, cv2.LINE_AA)


def _draw_limb_heatmap_centers(canvas, limb_heatmap, rel_thresh=0.35, min_area=5):
    """
    在肢体热度图叠加图上画每个肢体通道的中心点。

    输入 limb_heatmap 必须是 [C, H, W]。
    中心点估计方法：最大连通域 + 加权 PCA 主轴端点中点。
    颜色：左侧肢体蓝色十字，右侧肢体绿色十字，中轴肢体黄色十字。
    """
    if canvas is None or limb_heatmap is None:
        return canvas

    hm = limb_heatmap.detach().float().cpu() if torch.is_tensor(limb_heatmap) else limb_heatmap
    if torch.is_tensor(hm):
        hm = hm.numpy()
    hm = np.asarray(hm, dtype=np.float32)

    if hm.ndim != 3:
        return canvas

    img_h, img_w = canvas.shape[:2]
    num_limbs, hm_h, hm_w = hm.shape
    if hm_h <= 0 or hm_w <= 0:
        return canvas

    x_scale = float(img_w) / float(hm_w)
    y_scale = float(img_h) / float(hm_h)

    for limb_id in range(num_limbs):
        center_hm, info = _estimate_limb_center_from_heatmap_pca(
            hm[limb_id],
            rel_thresh=rel_thresh,
            min_area=min_area,
        )
        if center_hm is None:
            continue

        cx_hm, cy_hm = center_hm
        cx_img = cx_hm * x_scale
        cy_img = cy_hm * y_scale

        color = _limb_center_color(limb_id)
        _draw_cross_marker(canvas, cx_img, cy_img, color, size=7, thickness=2)

    return canvas

def _make_hconcat(images, pad=8):
    """
    横向拼接若干张图，不同尺寸时按最大高度补边。
    """
    valid = [im for im in images if im is not None]
    if len(valid) == 0:
        return None

    max_h = max(im.shape[0] for im in valid)
    padded = []
    for im in valid:
        h, w = im.shape[:2]
        if h < max_h:
            bottom = max_h - h
            im = cv2.copyMakeBorder(im, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        padded.append(im)
        if pad > 0:
            padded.append(np.full((max_h, pad, 3), 30, dtype=np.uint8))
    if pad > 0:
        padded = padded[:-1]

    return cv2.hconcat(padded)


def _pad_to_width(img, width, value=(30, 30, 30)):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w >= width:
        return img
    return cv2.copyMakeBorder(img, 0, 0, 0, int(width - w), cv2.BORDER_CONSTANT, value=value)


def _make_vconcat(images, pad=8):
    """
    纵向拼接若干张图，不同宽度时按最大宽度补边。
    """
    valid = [im for im in images if im is not None]
    if len(valid) == 0:
        return None

    max_w = max(im.shape[1] for im in valid)
    padded = []
    for im in valid:
        im = _pad_to_width(im, max_w)
        padded.append(im)
        if pad > 0:
            padded.append(np.full((pad, max_w, 3), 30, dtype=np.uint8))
    if pad > 0:
        padded = padded[:-1]

    return cv2.vconcat(padded)


def _make_score_panel(scores, height, width=250, title="Net1 easy scores"):
    """
    生成独立置信度信息面板，不把分数叠加到三张姿态图中。
    """
    panel_h = int(max(height, 260))
    panel_w = int(width)
    panel = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (90, 90, 90), 1)
    cv2.putText(panel, title, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (6, 34), (panel_w - 6, 34), (90, 90, 90), 1)

    if scores is None:
        return panel

    scores = np.asarray(scores).reshape(-1)

    col_x = [8, panel_w // 2 + 4]
    start_y = 58
    row_gap = 24

    for jid in range(min(NUM_KPTS, len(scores))):
        col = 0 if jid < 8 else 1
        row = jid if jid < 8 else jid - 8

        x = col_x[col]
        y = start_y + row * row_gap

        name = JOINT_NAMES[jid] if jid < len(JOINT_NAMES) else str(jid)
        score = float(scores[jid])
        text = "{:02d} {:>6s}: {:.3f}".format(jid, name, score)

        if score >= 0.5:
            color = (230, 230, 230)
        elif score >= 0.2:
            color = (170, 220, 255)
        else:
            color = (140, 140, 140)

        cv2.putText(panel, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    y2 = start_y + 8 * row_gap + 16
    cv2.line(panel, (6, y2), (panel_w - 6, y2), (70, 70, 70), 1)
    cv2.putText(panel, "Only easy(Net1) scores are shown.", (8, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

    return panel


def _get_batch_image_names(meta, batch_size):
    """
    从 DataLoader collate 后的 meta 中取出当前 batch 对应的原始无标签图片名。

    Mixed_MPII_AI_Dataset 返回的 meta 通常是 [labeled_meta, unlabeled_meta]，
    这里优先使用 meta[1]['image']。如果结构不匹配，则回退到通用文件名，
    只影响 debug 图片命名，不影响训练数据、损失或梯度。
    """
    names = None

    try:
        if isinstance(meta, (list, tuple)) and len(meta) > 1 and isinstance(meta[1], dict):
            names = meta[1].get('image', None)
        elif isinstance(meta, dict):
            names = meta.get('image', None)
    except Exception:
        names = None

    if names is None:
        return ["sample_{:06d}.jpg".format(i) for i in range(batch_size)]

    if isinstance(names, str):
        names = [names]
    elif isinstance(names, (list, tuple)):
        names = list(names)
    else:
        try:
            names = list(names)
        except Exception:
            names = []

    out = []
    for i in range(batch_size):
        if i < len(names):
            name = str(names[i])
            name = os.path.basename(name)
            if name == "":
                name = "sample_{:06d}.jpg".format(i)
        else:
            name = "sample_{:06d}.jpg".format(i)

        # Windows/Unix 文件名安全处理；正常图片名不会被改变。
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        stem, ext = os.path.splitext(name)
        if ext == "":
            name = name + ".jpg"
        out.append(name)

    return out


def _resolve_save_path(save_dir, file_name):
    """
    按原始图片文件名保存；如果同一 epoch 内出现重名，追加 _dupXX，避免覆盖。
    """
    save_path = os.path.join(save_dir, file_name)
    if not os.path.exists(save_path):
        return save_path

    stem, ext = os.path.splitext(file_name)
    if ext == "":
        ext = ".jpg"
    for dup_idx in range(1, 10000):
        cand = os.path.join(save_dir, "{}_dup{:02d}{}".format(stem, dup_idx, ext))
        if not os.path.exists(cand):
            return cand
    return os.path.join(save_dir, "{}_{}{}".format(stem, int(time.time() * 1000), ext))


def _extract_last_integer_from_filename(file_name):
    """
    从原始图片文件名中提取最后一组连续数字。

    例如：
        000010.jpg -> 10
        frame_000120_left.jpg -> 120
    如果文件名中没有数字，则返回 None。
    """
    base = os.path.basename(str(file_name))
    stem, _ = os.path.splitext(base)
    matches = re.findall(r"\d+", stem)
    if len(matches) == 0:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _is_image_number_multiple_of_10(file_name):
    """
    判断原始图片名中的数字是否为 10 的整数倍。
    没有数字的图片名默认不保存，避免误保存。
    """
    number = _extract_last_integer_from_filename(file_name)
    if number is None:
        return False
    return number % 10 == 0


def _unpack_pose_dual_output(output):
    """
    兼容 pose_dual 的 single path 和 multi path 输出。
    """
    if not isinstance(output, (list, tuple)) or len(output) != 9:
        return None

    sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1, cons_ht2, out_dic = output

    if isinstance(cons_ht1, (list, tuple)) and isinstance(cons_ht2, (list, tuple)):
        cons_ht1_list = list(cons_ht1)
        cons_ht2_list = list(cons_ht2)
    else:
        cons_ht1_list = [cons_ht1]
        cons_ht2_list = [cons_ht2]

    if out_dic is None:
        out_dic = {}

    return {
        "sup_ht1": sup_ht1,
        "sup_ht2": sup_ht2,
        "unsup_ht1": unsup_ht1,
        "unsup_ht2": unsup_ht2,
        "unsup_ht_trans1": unsup_ht_trans1,
        "unsup_ht_trans2": unsup_ht_trans2,
        "cons_ht1_list": cons_ht1_list,
        "cons_ht2_list": cons_ht2_list,
        "easy_limb": None,
        "hard_limb_list": [],
        "out_dic": out_dic,
    }




def _unpack_pose_hetero_limb_output(output):
    """
    兼容新增的 pose_hetero_limb 输出。
    使用关节点网络的 easy teacher 和 hard student 预测进行可视化。
    """
    if not isinstance(output, dict):
        return None

    if "unsup_kp_easy" not in output or "hard_kp_list" not in output:
        return None

    hard_kp_list = output.get("hard_kp_list", None)
    if hard_kp_list is None:
        return None
    if not isinstance(hard_kp_list, (list, tuple)):
        hard_kp_list = [hard_kp_list]

    out_dic = output.get("out_dic", {})
    if out_dic is None:
        out_dic = {}

    hard_views = out_dic.get("hard_views", None)
    if hard_views is None:
        return None
    if not isinstance(hard_views, (list, tuple)):
        hard_views = [hard_views]

    hard_limb_list = output.get("hard_limb_list", None)
    if hard_limb_list is None:
        hard_limb_list = []
    elif not isinstance(hard_limb_list, (list, tuple)):
        hard_limb_list = [hard_limb_list]
    else:
        hard_limb_list = list(hard_limb_list)

    easy_x = out_dic.get("easy_x", None)
    if easy_x is None:
        # 若模型没有返回 easy_x，则退化为第一路 hard 图像作为背景图，避免调试函数报错。
        easy_x = hard_views[0] if len(hard_views) > 0 else None

    return {
        "sup_ht1": None,
        "sup_ht2": None,
        "unsup_ht1": None,
        "unsup_ht2": None,
        # easy_x 使用 easy view 图像，因此这里必须使用 easy 坐标系下的 teacher 预测。
        # kp_teacher_hard 可能已经被 hard-view 变换过，不能叠加到 easy_x 上。
        "unsup_ht_trans1": output.get("unsup_kp_easy", None),
        "unsup_ht_trans2": None,
        "cons_ht1_list": list(hard_kp_list),
        "cons_ht2_list": list(hard_kp_list),
        "easy_limb": output.get("unsup_limb_easy", None),
        "hard_limb_list": hard_limb_list,
        "out_dic": {
            "easy_x1": easy_x,
            "cons_x2_list": list(hard_views),
        },
    }

def debug_visualize_pose_dual_easy_hard(
        output,
        meta=None,
        save_dir="debug_easy_hard_pose",
        max_images=2,
        save_every=200,
        enable=True,
        tensor_is_rgb=True,
        only_rank0=True,
        conf_thr_draw=0.0):
    """
    保存 PoseDual 半监督训练中 easy teacher 与 hard student 的预测对比图。

    当 max_images=None 或 max_images<=0 时，保存当前 batch 的全部无标签样本。
    如果传入 meta，则优先使用 meta[1]['image'] 中的原始无标签图片名命名。
    本函数只 detach 后做 debug 可视化，不参与模型参数更新。
    """
    if not enable:
        return

    if only_rank0 and not _is_main_process_from_env():
        return

    if not hasattr(debug_visualize_pose_dual_easy_hard, "_counter"):
        debug_visualize_pose_dual_easy_hard._counter = 0
    debug_visualize_pose_dual_easy_hard._counter += 1
    counter = debug_visualize_pose_dual_easy_hard._counter

    if save_every is not None and save_every > 1:
        if counter % int(save_every) != 1:
            return

    unpacked = _unpack_pose_dual_output(output)
    if unpacked is None:
        unpacked = _unpack_pose_hetero_limb_output(output)
    if unpacked is None:
        return

    easy_ht = unpacked["unsup_ht_trans1"]
    hard_ht_list = unpacked["cons_ht2_list"]
    easy_limb = unpacked.get("easy_limb", None)
    hard_limb_list = unpacked.get("hard_limb_list", [])
    if hard_limb_list is None:
        hard_limb_list = []
    if not isinstance(hard_limb_list, (list, tuple)):
        hard_limb_list = [hard_limb_list]
    out_dic = unpacked["out_dic"]

    if easy_ht is None or len(hard_ht_list) == 0:
        return

    cons_x2_list = out_dic.get("cons_x2_list", None)

    if cons_x2_list is None or len(cons_x2_list) == 0:
        if "unsup_x_trans_2" in out_dic:
            hard_img_list = [out_dic["unsup_x_trans_2"]]
        else:
            return
    else:
        hard_img_list = list(cons_x2_list)

    if "easy_x1" in out_dic:
        easy_img_tensor = out_dic["easy_x1"]
    elif "unsup_x_trans" in out_dic:
        easy_img_tensor = out_dic["unsup_x_trans"]
    else:
        easy_img_tensor = hard_img_list[0]

    num_hard_paths = min(2, len(hard_ht_list), len(hard_img_list))
    if num_hard_paths <= 0:
        return

    batch_size = int(easy_ht.shape[0])
    image_names = _get_batch_image_names(meta, batch_size)

    # 只对“原图片名中的数字是 10 的整数倍”的无标签样本计算并保存 debug 图。
    # 这里先筛选 index，再计算 heatmap 坐标，避免对不保存的样本做额外可视化计算。
    selected_indices = [
        b for b in range(batch_size)
        if b < len(image_names) and _is_image_number_multiple_of_10(image_names[b])
    ]

    if len(selected_indices) == 0:
        return

    os.makedirs(save_dir, exist_ok=True)

    if max_images is not None:
        max_images = int(max_images)
        if max_images > 0:
            selected_indices = selected_indices[:max_images]

    _, _, img_h, img_w = easy_img_tensor.shape

    easy_pts, easy_scores = _heatmap_to_points_and_scores(easy_ht[selected_indices], img_w, img_h)
    hard_pts_scores = [
        _heatmap_to_points_and_scores(hard_ht_list[p][selected_indices], img_w, img_h)
        for p in range(num_hard_paths)
    ]

    for local_idx, b in enumerate(selected_indices):
        easy_img = _tensor_img_to_bgr(easy_img_tensor[b], tensor_is_rgb=tensor_is_rgb)
        if easy_img is None:
            continue

        easy_vis = _draw_pose(
            easy_img,
            easy_pts[local_idx],
            easy_scores[local_idx],
            title="Net1 easy teacher",
            conf_thr_draw=conf_thr_draw,
            line_thickness=1,
            joint_radius=2,
            text_scale=0.35,
            joint_text=True,
        )

        joint_panels = [easy_vis]
        limb_panels = []

        if easy_limb is not None and torch.is_tensor(easy_limb) and b < easy_limb.shape[0]:
            easy_limb_vis = _limb_heatmap_to_overlay(
                easy_img,
                easy_limb[b],
                title="Easy teacher limb heatmap",
            )
            if easy_limb_vis is not None:
                limb_panels.append(easy_limb_vis)

        for p in range(num_hard_paths):
            hard_img = _tensor_img_to_bgr(hard_img_list[p][b], tensor_is_rgb=tensor_is_rgb)
            if hard_img is None:
                continue

            hard_pts, hard_scores = hard_pts_scores[p]
            if p == 0:
                path_name = "Hard path0 joints"
                limb_name = "Hard path0 limb heatmap"
            elif p == 1:
                path_name = "Hard path1 joints"
                limb_name = "Hard path1 limb heatmap"
            else:
                path_name = "Hard path{} joints".format(p)
                limb_name = "Hard path{} limb heatmap".format(p)

            hard_vis = _draw_pose(
                hard_img,
                hard_pts[local_idx],
                hard_scores[local_idx],
                title=path_name,
                conf_thr_draw=conf_thr_draw,
                line_thickness=1,
                joint_radius=2,
                text_scale=0.35,
                joint_text=True,
            )
            joint_panels.append(hard_vis)

            if p < len(hard_limb_list):
                limb_hm = hard_limb_list[p]
                if limb_hm is not None and torch.is_tensor(limb_hm) and b < limb_hm.shape[0]:
                    hard_limb_vis = _limb_heatmap_to_overlay(
                        hard_img,
                        limb_hm[b],
                        title=limb_name,
                    )
                    if hard_limb_vis is not None:
                        limb_panels.append(hard_limb_vis)

        joint_row = _make_hconcat(joint_panels, pad=8)
        if joint_row is None:
            continue

        # 对 pose_hetero_limb：上排为关节点骨架图，下排为对应图像上的肢体热度图。
        # 对 pose_dual：没有肢体热图输出时，保持原来的单排显示。
        if len(limb_panels) > 0:
            limb_row = _make_hconcat(limb_panels, pad=8)
            merged_imgs = _make_vconcat([joint_row, limb_row], pad=8)
        else:
            merged_imgs = joint_row

        if merged_imgs is None:
            continue

        score_panel = _make_score_panel(easy_scores[local_idx], merged_imgs.shape[0], width=250)
        merged = _make_hconcat([merged_imgs, score_panel], pad=8)
        if merged is None:
            continue

        save_name = image_names[b] if b < len(image_names) else "sample_{:06d}.jpg".format(b)
        save_path = _resolve_save_path(save_dir, save_name)
        cv2.imwrite(save_path, merged)
