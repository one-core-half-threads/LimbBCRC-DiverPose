# ------------------------------------------------------------------------------
# Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np



########################################################################

# YOCO(ICML2022) You Only Cut Once: Boosting Data Augmentation with a Single Cut
# https://github.com/JunlinHan/YOCO
def random_YOCO(images, images_aux):  # images or images_aux is processed by RandAug
    N,_,w,h = images.shape
    
    for i in range(N):
        if torch.rand(1) > 0.5:
            # images[i, :, 0:int(w/2), :] = images_aux[i, :, 0:int(w/2), :]  # also right
            images[i, :, int(w/2):w, :] = images_aux[i, :, int(w/2):w, :]
        else:
            # images[i, :, :, 0:int(h/2)] = images_aux[i, :, :, 0:int(h/2)]  # also right
            images[i, :, :, int(h/2):h] = images_aux[i, :, :, int(h/2):h]
    return images

########################################################################
    
# arxiv2017.08 (Cutout) Improved Regularization of Convolutional Neural Networks with Cutout
# https://github.com/uoguelph-mlrg/Cutout
def random_cutout(image, MASK_HOLES_NUM=4):
    N,_,width,height = image.shape
    center_x = torch.randint(0,width,(N,MASK_HOLES_NUM)).int().cuda()
    center_y = torch.randint(0,height,(N,MASK_HOLES_NUM)).int().cuda()
    size = torch.randint(10,20,(N,MASK_HOLES_NUM,2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0],0,width)
    y0 = torch.clamp_(center_y-size[...,1],0,height)

    x1 = torch.clamp_(center_x+size[...,0],0,width)
    y1 = torch.clamp_(center_y+size[...,1],0,height)

    for i in range(N):
        for j in range(MASK_HOLES_NUM):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = 0
    return image

    
# ICLR2018 Mixup - Beyond Empirical Risk Minimization
# https://github.com/facebookresearch/mixup-cifar10
def random_mixup(image):
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)  # lam in (0 ,1)
    
    N,_,width,height = image.shape
    rand_index = torch.randperm(N).cuda()
    image_rand = image[rand_index]
    
    lam = 1 - lam * 0.25  # we want the lam in (0.75, 1), and keep most features of the original image
    
    # for i in range(N):
        # mixed_image = lam * image[i, ...] + (1 - lam) * image_rand[i, ...]
        # image[i, ...] = mixed_image
    # return image
    
    mixed_image = lam * image + (1 - lam) * image_rand
    return mixed_image


# ICCV2019 CutMix - Regularization Strategy to Train Strong Classifiers With Localizable Features
# https://github.com/clovaai/CutMix-PyTorch

########################################################################
def random_cutmix(image, MASK_HOLES_NUM=2):
    N,_,width,height = image.shape
    center_x = torch.randint(0,width,(N,MASK_HOLES_NUM)).int().cuda()
    center_y = torch.randint(0,height,(N,MASK_HOLES_NUM)).int().cuda()
    size = torch.randint(10,20,(N,MASK_HOLES_NUM,2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0],0,width)
    y0 = torch.clamp_(center_y-size[...,1],0,height)

    x1 = torch.clamp_(center_x+size[...,0],0,width)
    y1 = torch.clamp_(center_y+size[...,1],0,height)
    
    rand_index = torch.randperm(N).cuda()
    image_rand = image[rand_index]

    for i in range(N):
        for j in range(MASK_HOLES_NUM):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
    return image


# ICCV2021 An Empirical Study of the Collapsing Problem in Semi-Supervised 2D Human Pose Estimation
# https://github.com/xierc/Semi_Human_Pose (Joint Cutout)
def mask_joint(image,joints,MASK_JOINT_NUM=4):
    ## N,J,2 joints
    N,J = joints.shape[:2]
    _,_,width,height = image.shape
    re_joints = joints[:,:,:2] + torch.randn((N,J,2)).cuda()*10
    re_joints = re_joints.int()
    size = torch.randint(10,20,(N,J,2)).int().cuda()

    x0 = re_joints[:,:,0]-size[:,:,0]
    y0 = re_joints[:,:,1]-size[:,:,1]

    x1 = re_joints[:,:,0]+size[:,:,0]
    y1 = re_joints[:,:,1]+size[:,:,1]

    torch.clamp_(x0,0,width)
    torch.clamp_(x1,0,width)
    torch.clamp_(y0,0,height)
    torch.clamp_(y1,0,height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(J, MASK_JOINT_NUM)
        for j in ind:
            image[i,:,y0[i,j]:y1[i,j],x0[i,j]:x1[i,j]] = 0
    return image


# CVPR2023, Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction Module
# https://github.com/hlz0606/SSPCM/blob/main/SSPCM/lib/models/my_pose_triple.py#L836
# Semi-supervised Cut-Occlude based on pseudo keypoint perception (SSCO). See Figure 5 in the original paper.
def keypoint_cutmix_perception(image, joints, MASK_JOINT_NUM=2):
    """Keypoint-aware CutMix around predicted joint locations.

    Args:
        image: Tensor [N, C, H, W].
        joints: Tensor [N, J, 2] in input-image coordinates, where
                joints[..., 0] is x and joints[..., 1] is y.
        MASK_JOINT_NUM: number of joints to use per sample.

    Notes:
        The original imported SSPCM/MultiAugs-style code used
        re_joints[..., 1] as x and re_joints[..., 0] as y.  In this project,
        get_max_preds_tensor returns coordinates in the standard order
        [x, y], and mask_joint() also follows [x, y].  Therefore JO must use
        re_joints[..., 0] for x and re_joints[..., 1] for y.  Otherwise the
        CutMix patch is centered near (y, x), i.e., far away from the intended
        joint for most non-square-symmetric poses.
    """
    image_tmp = image.clone()

    if image is None or image.dim() != 4:
        return image

    # image layout is [N, C, H, W].
    N, J = joints.shape[:2]
    _, _, height, width = image.shape
    device = image.device
    dtype = joints.dtype if torch.is_floating_point(joints) else torch.float32

    # Add small random localization noise as in the original implementation,
    # but keep tensors on the same device as the input image.
    noise = torch.randn((N, J, 2), device=device, dtype=dtype) * 10.0
    re_joints = (joints[:, :, :2].to(device=device, dtype=dtype) + noise).int()
    size = torch.randint(10, 20, (N, J, 2), device=device).int()

    # Correct coordinate convention: [..., 0] is x, [..., 1] is y.
    center_x = torch.clamp(re_joints[:, :, 0], 0, width - 1)
    center_y = torch.clamp(re_joints[:, :, 1], 0, height - 1)

    x0 = torch.clamp(center_x - size[:, :, 0], 0, width)
    x1 = torch.clamp(center_x + size[:, :, 0], 0, width)
    y0 = torch.clamp(center_y - size[:, :, 1], 0, height)
    y1 = torch.clamp(center_y + size[:, :, 1], 0, height)

    for i in range(N):
        if J <= 0:
            continue
        num = min(int(MASK_JOINT_NUM), int(J))
        ind = np.random.choice(J, num, replace=False if num <= J else True)
        ind_2 = np.random.choice(J, num, replace=False if num <= J else True)
        img_id = np.random.randint(0, N)

        for idx in range(len(ind)):
            j = int(ind[idx])
            j2 = int(ind_2[idx])

            # Source patch around joint j2 in image img_id.
            sx0 = int(x0[img_id, j2].item())
            sx1 = int(x1[img_id, j2].item())
            sy0 = int(y0[img_id, j2].item())
            sy1 = int(y1[img_id, j2].item())
            scx = int(center_x[img_id, j2].item())
            scy = int(center_y[img_id, j2].item())

            if sx1 <= sx0 or sy1 <= sy0:
                continue

            # Place a patch with the same source-side radii around joint j in
            # image i. If the destination is clipped by an image boundary, crop
            # the source by the same amount so source/target shapes match.
            dcx = int(center_x[i, j].item())
            dcy = int(center_y[i, j].item())

            left = scx - sx0
            right = sx1 - scx
            top = scy - sy0
            bottom = sy1 - scy

            dx0_raw = dcx - left
            dx1_raw = dcx + right
            dy0_raw = dcy - top
            dy1_raw = dcy + bottom

            dx0 = max(0, dx0_raw)
            dx1 = min(width, dx1_raw)
            dy0 = max(0, dy0_raw)
            dy1 = min(height, dy1_raw)

            # Adjust source crop if the destination crop has been clipped.
            sx0_adj = sx0 + (dx0 - dx0_raw)
            sx1_adj = sx1 - (dx1_raw - dx1)
            sy0_adj = sy0 + (dy0 - dy0_raw)
            sy1_adj = sy1 - (dy1_raw - dy1)

            # Final safety clamp and shape check.
            sx0_adj = max(0, min(width, sx0_adj))
            sx1_adj = max(0, min(width, sx1_adj))
            sy0_adj = max(0, min(height, sy0_adj))
            sy1_adj = max(0, min(height, sy1_adj))

            if dx1 <= dx0 or dy1 <= dy0 or sx1_adj <= sx0_adj or sy1_adj <= sy0_adj:
                continue

            dst = image[i, :, dy0:dy1, dx0:dx1]
            src = image_tmp[img_id, :, sy0_adj:sy1_adj, sx0_adj:sx1_adj]
            if dst.shape != src.shape:
                # This should be rare after the boundary adjustment above; keep
                # a safe fallback instead of raising during training.
                min_h = min(dst.shape[-2], src.shape[-2])
                min_w = min(dst.shape[-1], src.shape[-1])
                if min_h <= 0 or min_w <= 0:
                    continue
                image[i, :, dy0:dy0 + min_h, dx0:dx0 + min_w] = src[:, :min_h, :min_w]
            else:
                image[i, :, dy0:dy1, dx0:dx1] = src

    return image

