# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from core.inference import get_max_preds, get_max_preds_tensor
from utils.limb_heatmap import MPII_LIMB_PAIRS
from utils.limb_heatmap import build_limb_pairs
from utils.limb_heatmap import get_mpii_parent_ids


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.cfg = cfg

    def forward(self, output, target, target_weight=None, meta=None, unsup_flag=False):
        batch_size = output.size(0)
        num_joints = output.size(1)
        height = output.size(2)
        width = output.size(3)
        
        if self.cfg.CONF_MASKING and unsup_flag:
            preds, maxvals = get_max_preds_tensor(target.clone())  # unsup_ht or unsup_ht_trans1 or unsup_ht_trans2
            preds_mask = torch.greater(maxvals, self.cfg.CONF_THRESH).float()  # shape (batch_size, num_joints)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
          
        '''
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        '''
        
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()  # shape (batch_size, img_w*img_h)
            heatmap_gt = heatmaps_gt[idx].squeeze()  # shape (batch_size, img_w*img_h)
            if self.use_target_weight:
                hm_pd = heatmap_pred.mul(target_weight[:, idx])
                hm_gt = heatmap_gt.mul(target_weight[:, idx])
            else:
                hm_pd, hm_gt = heatmap_pred, heatmap_gt
                
            if self.cfg.SHARPEN_PRED and unsup_flag:
                hm_pd = F.softmax(hm_pd, dim=1)
                hm_gt = F.softmax(hm_gt, dim=1)
            
            if self.cfg.CONF_MASKING and unsup_flag:
                # print(hm_pd.shape, hm_gt.shape, preds_mask.shape, target_weight.shape)
                hm_pd = hm_pd.mul(torch.unsqueeze(preds_mask[:, idx], 1))
                hm_gt = hm_gt.mul(torch.unsqueeze(preds_mask[:, idx], 1))
                
            loss += 0.5 * self.criterion(hm_pd, hm_gt)
        
        return loss / num_joints


class PoseCoLoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(PoseCoLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight, cfg)

        self.use_target_weight = use_target_weight
        
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = int(cfg.MODEL.NUM_JOINTS)
        self.target_type = 'gaussian'

        self.cfg = cfg

    def forward(self, output, target, target_weight, meta):
        if type(target)==list:
            sup_target, unsup_target = target
            sup_target_weight, unsup_target_weight = target_weight
            sup_meta, upsup_meta = meta
        else:
            sup_target = target
            sup_target_weight = target_weight
            sup_meta = meta

        batch_size, joint_num, ht_height, ht_width = sup_target.shape
        pseudo_target = 0
        
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1, cons_ht2, out_dic = output
        else:
            sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1_list, cons_ht2_list, out_dic = output

        batch_size = sup_ht1.size(0)
        num_joints = sup_ht1.size(1)   

        loss_pose = 0.5*self.mse_criterion(sup_ht1, sup_target, sup_target_weight)
        loss_pose += 0.5*self.mse_criterion(sup_ht2, sup_target, sup_target_weight)
        
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            loss_cons = self.mse_criterion(cons_ht1, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
            loss_cons += self.mse_criterion(cons_ht2, unsup_ht_trans1.detach(),  unsup_target_weight, unsup_flag=True)
        else:
            loss_cons = 0
            if self.cfg.HEAT_FUSION:
                cons_ht1_avg = torch.stack(cons_ht1_list, dim=0).sum(0) / len(cons_ht1_list)
                loss_cons += self.mse_criterion(cons_ht1_avg, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
                cons_ht2_avg = torch.stack(cons_ht2_list, dim=0).sum(0) / len(cons_ht2_list)
                loss_cons += self.mse_criterion(cons_ht2_avg, unsup_ht_trans1.detach(), unsup_target_weight, unsup_flag=True)
            else:
                for cons_ht in cons_ht1_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht_trans2.detach(), unsup_target_weight, unsup_flag=True)
                for cons_ht in cons_ht2_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht_trans1.detach(), unsup_target_weight, unsup_flag=True)
                
        pseudo_target = [unsup_ht_trans2.detach().cpu(), unsup_ht_trans1.detach().cpu()]

        loss = loss_pose + loss_cons 
        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
        }

        return loss, loss_dic, pseudo_target


def _to_device_tensor(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True).float()
    return torch.as_tensor(value, device=device).float()


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _cfg_parent_ids(cfg):
    parent_ids = getattr(getattr(cfg, 'HETERO_LIMB', {}), 'PARENT_IDS', None)
    if parent_ids is None:
        parent_ids = get_mpii_parent_ids()
    return [int(x) for x in parent_ids]


def _get_meta_value(meta, key, default=None):
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def _limb_visibility_weight_from_meta(meta, limb_pairs, batch_size, device, dtype):
    """Build [B, L, 1] pseudo-limb weights from endpoint visibility.

    Build endpoint-visibility weights when explicitly needed by other experiments.
    The underwater hetero-limb training path does NOT use this for K->B, because
    both labeled and unlabeled frogman datasets are in the same MPII-style
    16-joint space and every loss is computed in that 16-joint space.
    """
    joints_vis = _get_meta_value(meta, 'joints_vis', None)
    num_limbs = len(limb_pairs)
    if joints_vis is None:
        return torch.ones(batch_size, num_limbs, 1, device=device, dtype=dtype)

    if not torch.is_tensor(joints_vis):
        joints_vis = torch.as_tensor(joints_vis, device=device)
    else:
        joints_vis = joints_vis.to(device=device, non_blocking=True)
    joints_vis = joints_vis.float()

    if joints_vis.dim() == 2:
        joints_vis = joints_vis.unsqueeze(-1)
    if joints_vis.size(0) != batch_size:
        return torch.ones(batch_size, num_limbs, 1, device=device, dtype=dtype)

    endpoint_vis = joints_vis[:, :, 0] > 0
    pairs = torch.as_tensor(limb_pairs, device=device, dtype=torch.long)
    valid = endpoint_vis[:, pairs[:, 0]] & endpoint_vis[:, pairs[:, 1]]
    return valid.to(dtype=dtype).unsqueeze(-1)


def soft_argmax_2d(heatmaps, beta=50.0):
    batch_size, num_joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch_size, num_joints, -1)
    prob = F.softmax(flat * float(beta), dim=-1)

    xs = torch.linspace(0, width - 1, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0, height - 1, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs)
    coord_x = (prob * grid_x.reshape(1, 1, -1)).sum(dim=-1)
    coord_y = (prob * grid_y.reshape(1, 1, -1)).sum(dim=-1)
    return torch.stack([coord_x, coord_y], dim=-1)


def decode_argmax_points(heatmaps):
    batch_size, num_joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch_size, num_joints, -1)
    conf, idx = flat.max(dim=-1)
    x = (idx % width).float()
    y = (idx // width).float()
    return torch.stack([x, y], dim=-1), conf


def generate_limb_heatmaps_from_points(
    points,
    confidences,
    limb_pairs,
    height,
    width,
    sigma=2.0,
    truncate=3.0,
):
    device = points.device
    dtype = points.dtype
    pairs = torch.as_tensor(limb_pairs, device=device, dtype=torch.long)
    p1 = points[:, pairs[:, 0], :]
    p2 = points[:, pairs[:, 1], :]
    c1 = confidences[:, pairs[:, 0]].clamp(0.0, 1.0)
    c2 = confidences[:, pairs[:, 1]].clamp(0.0, 1.0)
    amplitude = torch.sqrt(c1 * c2).to(dtype)

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs)
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, height, width, 2)

    p1_exp = p1[:, :, None, None, :]
    vec = (p2 - p1)[:, :, None, None, :]
    length_sq = (vec * vec).sum(dim=-1).clamp_min(1e-6)
    ap = grid - p1_exp
    t = (ap * vec).sum(dim=-1) / length_sq
    t = t.clamp(0.0, 1.0)
    closest = p1_exp + t[..., None] * vec
    dist_sq = ((grid - closest) ** 2).sum(dim=-1)

    sigma = max(float(sigma), 1e-6)
    radius = sigma * max(float(truncate), 1.0)
    target = torch.exp(-dist_sq / (2.0 * sigma * sigma))
    target = target * (dist_sq <= radius * radius).to(dtype)
    target = target * amplitude[:, :, None, None]
    return target


def limb_dice_coverage_loss(
    points_h,
    teacher_limb_map,
    limb_pairs,
    sigma=2.0,
    truncate=3.0,
    eps=1e-6,
):
    """Match the complete teacher limb support with a differentiable K limb map.

    ``points_h`` comes from soft-argmax on the hard-view keypoint student and
    therefore must keep its computation graph.  The easy-view limb prediction is
    a teacher target and is detached here, exactly like the original support
    loss.  Per-limb peak normalization makes Dice measure spatial coverage rather
    than heatmap amplitude.  No confidence/peak filtering is applied.
    """
    teacher = teacher_limb_map.detach().clamp(0.0, 1.0)
    batch_size, _, height, width = teacher.shape

    # Unit endpoint confidence prevents the keypoint student from reducing this
    # loss merely by lowering its heatmap amplitude.  Gradients can only improve
    # the soft-argmax endpoint geometry.
    endpoint_confidence = points_h.new_ones(batch_size, points_h.size(1))
    pred = generate_limb_heatmaps_from_points(
        points_h,
        endpoint_confidence,
        limb_pairs,
        height,
        width,
        sigma=sigma,
        truncate=truncate,
    )

    teacher_peak = teacher.amax(dim=(2, 3), keepdim=True).clamp_min(float(eps))
    pred_peak = pred.amax(dim=(2, 3), keepdim=True).clamp_min(float(eps))
    teacher = teacher / teacher_peak
    pred = pred / pred_peak

    intersection = (pred * teacher).sum(dim=(2, 3))
    # Both operands are continuous Gaussian limb maps rather than binary masks.
    # The squared-norm denominator gives Dice=1 (loss=0) when the two soft maps
    # are identical; an L1 denominator would keep a non-zero loss at equality.
    denominator = (
        pred.square().sum(dim=(2, 3))
        + teacher.square().sum(dim=(2, 3))
    )
    dice = (2.0 * intersection + float(eps)) / (denominator + float(eps))
    return (1.0 - dice).mean()


def limb_support_loss(
    points_h,
    teacher_limb_map,
    limb_pairs,
    num_samples=16,
    min_teacher_peak=0.05,
):
    teacher = teacher_limb_map.detach().clamp(0.0, 1.0)
    batch_size, num_limbs, height, width = teacher.shape
    pairs = torch.as_tensor(limb_pairs, device=points_h.device, dtype=torch.long)

    p1 = points_h[:, pairs[:, 0], :]
    p2 = points_h[:, pairs[:, 1], :]
    t = torch.linspace(
        0.0,
        1.0,
        int(num_samples),
        device=points_h.device,
        dtype=points_h.dtype,
    )
    samples = (1.0 - t)[None, None, :, None] * p1[:, :, None, :] + t[
        None, None, :, None
    ] * p2[:, :, None, :]

    x = samples[..., 0] / max(width - 1, 1) * 2.0 - 1.0
    y = samples[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).reshape(batch_size * num_limbs, -1, 1, 2)
    maps = teacher.reshape(batch_size * num_limbs, 1, height, width)
    values = F.grid_sample(
        maps,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    ).reshape(batch_size, num_limbs, -1)

    support = values.mean(dim=-1)
    peak = teacher.amax(dim=(2, 3))
    valid = (peak > float(min_teacher_peak)).float()
    if valid.sum().item() <= 0:
        return points_h.sum() * 0.0
    return (F.relu(peak - support) * valid).sum() / valid.sum().clamp_min(1.0)


class HeteroLimbCoLoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(HeteroLimbCoLoss, self).__init__()
        self.kp_mse = JointsMSELoss(use_target_weight, cfg)
        self.limb_mse = JointsMSELoss(use_target_weight, cfg)
        self.cfg = cfg
        self.parent_ids = _cfg_parent_ids(cfg)
        self.limb_pairs = build_limb_pairs(self.parent_ids, int(cfg.MODEL.NUM_JOINTS))
        self.kp_sup_weight = float(cfg.HETERO_LIMB.KP_SUP_WEIGHT)
        self.limb_sup_weight = float(cfg.HETERO_LIMB.LIMB_SUP_WEIGHT)
        self.kk_weight = float(cfg.HETERO_LIMB.KK_WEIGHT)
        self.bb_weight = float(cfg.HETERO_LIMB.BB_WEIGHT)
        self.b_to_k_weight = float(cfg.HETERO_LIMB.B_TO_K_WEIGHT)
        self.b_to_k_dice_weight = float(
            getattr(cfg.HETERO_LIMB, 'B_TO_K_DICE_WEIGHT', 0.25)
        )
        self.k_to_b_weight = float(cfg.HETERO_LIMB.K_TO_B_WEIGHT)
        self.warmup_epoch = int(cfg.HETERO_LIMB.WARMUP_EPOCH)
        self.softargmax_beta = float(cfg.HETERO_LIMB.SOFTARGMAX_BETA)
        self.limb_sample_num = int(cfg.HETERO_LIMB.LIMB_SAMPLE_NUM)
        self.min_teacher_peak = float(cfg.HETERO_LIMB.LIMB_TEACHER_MIN_PEAK)
        self.k_to_b_min_peak = float(getattr(cfg.HETERO_LIMB, 'K_TO_B_MIN_PEAK', self.min_teacher_peak))
        self.limb_sigma = float(getattr(cfg.DATASET, 'LIMB_HEATMAP_SIGMA', cfg.HETERO_LIMB.LIMB_HEATMAP_SIGMA))
        self.limb_truncate = float(getattr(cfg.DATASET, 'LIMB_HEATMAP_TRUNCATE', cfg.HETERO_LIMB.LIMB_HEATMAP_TRUNCATE))

    def _cross_task_scale(self, meta):
        if not isinstance(meta, dict):
            return 1.0
        epoch = int(meta.get('epoch', 0))
        return 0.0 if epoch < self.warmup_epoch else 1.0

    def _supervised_limb_target(self, sup_meta, device):
        if 'limb_target' not in sup_meta or 'limb_target_weight' not in sup_meta:
            raise KeyError(
                "Hetero limb training requires meta['limb_target'] and "
                "meta['limb_target_weight']. Enable HETERO_LIMB in the config."
            )
        limb_target = _to_device_tensor(sup_meta['limb_target'], device)
        limb_weight = _to_device_tensor(sup_meta['limb_target_weight'], device)
        return limb_target, limb_weight

    def forward(self, output, target, target_weight, meta):
        if not isinstance(output, dict):
            raise TypeError('HeteroLimbCoLoss expects dict output from pose_hetero_limb.')

        if isinstance(target, list):
            sup_target, unsup_target = target
            sup_target_weight, unsup_target_weight = target_weight
            sup_meta, unsup_meta = meta
        else:
            sup_target = target
            unsup_target = None
            sup_target_weight = target_weight
            unsup_target_weight = torch.ones_like(target_weight)
            sup_meta = meta
            unsup_meta = {}

        device = output['sup_kp'].device
        limb_target, limb_target_weight = self._supervised_limb_target(sup_meta, device)

        loss_sup_kp = self.kp_mse(
            output['sup_kp'], sup_target, sup_target_weight
        ) * self.kp_sup_weight
        loss_sup_limb = self.limb_mse(
            output['sup_limb'], limb_target, limb_target_weight
        ) * self.limb_sup_weight

        hard_kp_list = _as_list(output['hard_kp_list'])
        hard_limb_list = _as_list(output['hard_limb_list'])

        # Hard augmentations may provide per-view teacher heatmaps when they
        # include geometric transforms. When *_list is absent, keep the old behavior.
        kp_teacher_list = _as_list(output.get('kp_teacher_hard_list', output['kp_teacher_hard']))
        limb_teacher_list = _as_list(output.get('limb_teacher_hard_list', output['limb_teacher_hard']))
        if len(kp_teacher_list) == 1 and len(hard_kp_list) > 1:
            kp_teacher_list = kp_teacher_list * len(hard_kp_list)
        if len(limb_teacher_list) == 1 and len(hard_limb_list) > 1:
            limb_teacher_list = limb_teacher_list * len(hard_limb_list)
        if len(kp_teacher_list) != len(hard_kp_list):
            raise RuntimeError(
                'Number of keypoint teacher targets ({}) does not match hard views ({}).'.format(
                    len(kp_teacher_list), len(hard_kp_list)
                )
            )
        if len(limb_teacher_list) != len(hard_limb_list):
            raise RuntimeError(
                'Number of limb teacher targets ({}) does not match hard views ({}).'.format(
                    len(limb_teacher_list), len(hard_limb_list)
                )
            )
        kp_teacher_list = [x.detach() for x in kp_teacher_list]
        limb_teacher_list = [x.detach() for x in limb_teacher_list]

        # Lkk is an unlabeled teacher-student consistency loss.
        # Its target is K(Ie), not AIC/DiversUnlabeled annotation, so all 16
        # MPII channels (including root/spine) should participate by default.
        loss_kk = output['sup_kp'].new_tensor(0.0)
        for hard_kp, kp_teacher in zip(hard_kp_list, kp_teacher_list):
            unsup_kp_weight = torch.ones(
                kp_teacher.size(0), kp_teacher.size(1), 1,
                device=device, dtype=kp_teacher.dtype,
            )
            loss_kk = loss_kk + self.kp_mse(
                hard_kp, kp_teacher, unsup_kp_weight, unsup_flag=False
            )
        loss_kk = loss_kk * self.kk_weight

        cross_scale = self._cross_task_scale(sup_meta)
        # B-B is the limb-network counterpart of K-K: the aligned easy-view
        # limb prediction supervises each hard-view limb prediction. Reuse the
        # per-view lists so single-view, MULTI_AUGS, and REPEAT_AUG all follow
        # exactly the same pairing and geometry-alignment path as K-K.
        loss_bb = output['sup_kp'].new_tensor(0.0)
        if self.bb_weight > 0:
            for hard_limb, limb_teacher in zip(hard_limb_list, limb_teacher_list):
                unsup_limb_weight = torch.ones(
                    limb_teacher.size(0), limb_teacher.size(1), 1,
                    device=device, dtype=limb_teacher.dtype,
                )
                loss_bb = loss_bb + self.limb_mse(
                    hard_limb, limb_teacher, unsup_limb_weight, unsup_flag=False
                )
            loss_bb = loss_bb * self.bb_weight

        loss_b_to_k_support = output['sup_kp'].new_tensor(0.0)
        loss_b_to_k_dice = output['sup_kp'].new_tensor(0.0)
        if self.b_to_k_weight > 0 and cross_scale > 0:
            for hard_kp, limb_teacher in zip(hard_kp_list, limb_teacher_list):
                points_h = soft_argmax_2d(hard_kp, beta=self.softargmax_beta)
                loss_b_to_k_support = loss_b_to_k_support + limb_support_loss(
                    points_h,
                    limb_teacher,
                    self.limb_pairs,
                    num_samples=self.limb_sample_num,
                    min_teacher_peak=self.min_teacher_peak,
                )
                if self.b_to_k_dice_weight > 0:
                    loss_b_to_k_dice = loss_b_to_k_dice + limb_dice_coverage_loss(
                        points_h,
                        limb_teacher,
                        self.limb_pairs,
                        sigma=self.limb_sigma,
                        truncate=self.limb_truncate,
                    )
            loss_b_to_k_support = (
                loss_b_to_k_support * self.b_to_k_weight * cross_scale
            )
            loss_b_to_k_dice = (
                loss_b_to_k_dice
                * self.b_to_k_weight
                * self.b_to_k_dice_weight
                * cross_scale
            )
        # loss_b_to_k = loss_b_to_k_support + loss_b_to_k_dice
        loss_b_to_k = loss_b_to_k_dice

        loss_k_to_b = output['sup_kp'].new_tensor(0.0)
        if self.k_to_b_weight > 0 and cross_scale > 0:
            # K->B follows the paper's order exactly:
            # easy-view keypoint heatmaps -> pseudo-limb generation -> easy-to-hard alignment.
            kp_teacher_easy = output['unsup_kp_easy'].detach()
            teacher_to_hard_theta_list = _as_list(
                output['out_dic'].get(
                    'teacher_to_hard_theta_list',
                    output['out_dic']['teacher_to_hard_theta'],
                )
            )
            if len(teacher_to_hard_theta_list) == 1 and len(hard_limb_list) > 1:
                teacher_to_hard_theta_list = teacher_to_hard_theta_list * len(hard_limb_list)
            if len(teacher_to_hard_theta_list) != len(hard_limb_list):
                raise RuntimeError(
                    'Number of K->B alignment transforms ({}) does not match hard views ({}).'.format(
                        len(teacher_to_hard_theta_list), len(hard_limb_list)
                    )
                )

            with torch.no_grad():
                points_e, conf_e = decode_argmax_points(kp_teacher_easy)
                pseudo_limb_easy = generate_limb_heatmaps_from_points(
                    points_e,
                    conf_e,
                    self.limb_pairs,
                    kp_teacher_easy.size(2),
                    kp_teacher_easy.size(3),
                    sigma=self.limb_sigma,
                    truncate=self.limb_truncate,
                )

            for hard_limb, teacher_to_hard_theta in zip(
                hard_limb_list, teacher_to_hard_theta_list
            ):
                with torch.no_grad():
                    pseudo_grid = F.affine_grid(
                        teacher_to_hard_theta.to(
                            device=pseudo_limb_easy.device, dtype=torch.float32
                        ),
                        pseudo_limb_easy.size(),
                        align_corners=True,
                    ).float()
                    pseudo_limb = F.grid_sample(
                        pseudo_limb_easy.detach(), pseudo_grid, align_corners=True
                    )
                    # All underwater labeled/unlabeled samples are represented in the same
                    # MPII-style 16-joint output space. Therefore K->B must not mask
                    # limbs by AIC-14/source-joint visibility. Every pseudo limb is
                    # generated from the 16-channel teacher K(Ie). The only reliability
                    # filter here is the pseudo-limb peak threshold.
                    pseudo_peak = pseudo_limb.amax(dim=(2, 3), keepdim=False)
                    pseudo_weight = (pseudo_peak > self.k_to_b_min_peak).to(pseudo_limb.dtype).unsqueeze(-1)
                loss_k_to_b = loss_k_to_b + self.limb_mse(
                    hard_limb, pseudo_limb.detach(), pseudo_weight, unsup_flag=False
                )
            loss_k_to_b = loss_k_to_b * self.k_to_b_weight * cross_scale

        loss = loss_sup_kp + loss_sup_limb + loss_kk + loss_bb + loss_b_to_k + loss_k_to_b
        loss_dic = {
            'loss_sup_kp': loss_sup_kp.detach(),
            'loss_sup_limb': loss_sup_limb.detach(),
            'loss_kk': loss_kk.detach(),
            'loss_bb': loss_bb.detach(),
            'loss_b_to_k': loss_b_to_k.detach(),
            'loss_b_to_k_support': loss_b_to_k_support.detach(),
            'loss_b_to_k_dice': loss_b_to_k_dice.detach(),
            'loss_k_to_b': loss_k_to_b.detach(),
        }
        return loss, loss_dic

class PoseDisLoss(nn.Module):
    def __init__(self, use_target_weight, cfg=None):
        super(PoseDisLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight,cfg)

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = int(cfg.MODEL.NUM_JOINTS)
        self.target_type = 'gaussian'
        
        self.cfg = cfg


    def forward(self, output, target, target_weight, meta):
        # unpackage
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            sup_ht, _, unsup_ht, _, cons_ht, _ = output
        else:
            sup_ht, _, unsup_ht, _, cons_ht_list, _ = output

        batch_size, joint_num, ht_height, ht_width = sup_ht.shape

        sup_target, unsup_target = target
        sup_target_weight, unsup_target_weight = target_weight
        
        # JointsMSELoss of supervised sample
        # Loss Pose
        loss_pose = self.mse_criterion(sup_ht, sup_target, sup_target_weight)

        # preds, maxvals = get_max_preds_tensor(unsup_ht.detach())
     
        # JointsMSELoss of unsupervised sample
        # if not self.cfg.MULTI_AUGS:
        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            loss_cons = self.mse_criterion(cons_ht, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)
        else:
            loss_cons = 0
            if self.cfg.HEAT_FUSION:
                cons_ht_avg = torch.stack(cons_ht_list, dim=0).sum(0) / len(cons_ht_list)
                loss_cons += self.mse_criterion(cons_ht_avg, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)
            else:
                for cons_ht in cons_ht_list:
                    loss_cons += self.mse_criterion(cons_ht, unsup_ht.detach(), unsup_target_weight, unsup_flag=True)

        loss = loss_pose + loss_cons

        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
        }

        return loss, loss_dic
