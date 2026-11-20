from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference import get_max_preds_tensor
from utils.augments import keypoint_cutmix_perception
from utils.augments import mask_joint
from utils.augments import random_cutmix
from utils.augments import random_cutout
from utils.augments import random_mixup
from utils.foreground_recomposition import foreground_background_recomposition_aug

from .pose_hrnet import PoseHighResolutionNet


logger = logging.getLogger(__name__)


def _make_hrnet_cfg(cfg, model_cfg):
    net_cfg = copy.deepcopy(cfg)
    net_cfg.MODEL = copy.deepcopy(model_cfg)
    net_cfg.MODEL.NAME = 'pose_hrnet'
    net_cfg.MODEL.BACKBONE = 'hrnet'
    return net_cfg


def _heatmap(output):
    if isinstance(output, (tuple, list)):
        return output[-1]
    return output


def _identity_theta(batch_size, device):
    theta = torch.eye(2, 3, device=device, dtype=torch.float32)
    return theta.unsqueeze(0).repeat(batch_size, 1, 1)


def _to_homogeneous(theta):
    batch_size = theta.size(0)
    bottom = torch.tensor([0.0, 0.0, 1.0], device=theta.device, dtype=theta.dtype)
    bottom = bottom.view(1, 1, 3).repeat(batch_size, 1, 1)
    return torch.cat([theta, bottom], dim=1)


def _from_homogeneous(mat):
    return mat[:, :2, :]


def _compose_affine_theta(first_theta, second_theta):
    """Compose two affine_grid theta matrices.

    Both matrices follow affine_grid convention, mapping output normalized
    coordinates to input normalized coordinates.  If an image is first sampled
    with first_theta and then the result is sampled with second_theta, the
    composed mapping from final output coordinates to the original input is:
        first_theta @ second_theta.
    """
    first_h = _to_homogeneous(first_theta)
    second_h = _to_homogeneous(second_theta)
    return _from_homogeneous(first_h.bmm(second_h))


def _compose_teacher_to_hard_theta(teacher_theta, hard_theta):
    """Return theta mapping hard output coords to teacher coords.

    affine_grid theta maps output normalized coordinates to input normalized
    coordinates. If teacher image is generated from base by A_e and hard image
    from base by A_h, then a point y_h in hard output corresponds to base
    coordinate A_h y_h. The matching teacher coordinate y_e satisfies
    A_e y_e = A_h y_h, hence y_e = inv(A_e) A_h y_h.
    """
    teacher_h = _to_homogeneous(teacher_theta)
    hard_h = _to_homogeneous(hard_theta)
    return _from_homogeneous(torch.inverse(teacher_h).bmm(hard_h))


class HeteroLimbCotrain(nn.Module):
    def __init__(self, keypoint_net, limb_net, cfg):
        super(HeteroLimbCotrain, self).__init__()
        self.keypoint_net = keypoint_net
        self.limb_net = limb_net
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.cfg = cfg

    def get_batch_affine_transform(self, batch_size):
        batch_trans = []
        for _ in range(batch_size):
            scale = np.clip(
                np.random.randn() * self.scale_factor + 1,
                1 - self.scale_factor,
                1 + self.scale_factor,
            )
            rot = (
                np.clip(
                    np.random.randn() * self.rotation_factor,
                    -self.rotation_factor,
                    self.rotation_factor,
                )
                if random.random() <= 0.8
                else 0
            )
            batch_trans.append(cv2.getRotationMatrix2D((0, 0), rot, scale))
        return torch.from_numpy(np.stack(batch_trans, 0)).float()

    def _apply_named_aug(self, image, aug_name, preds=None, meta=None):
        image, _ = self._apply_named_aug_with_theta(image, aug_name, preds, meta)
        return image

    def _apply_named_aug_with_theta(self, image, aug_name, preds=None, meta=None):
        theta_identity = _identity_theta(image.shape[0], image.device)
        if aug_name == 'A30':
            return image, theta_identity
        if aug_name == 'A60':
            theta = self.get_batch_affine_transform(image.shape[0]).to(image.device)
            grid = F.affine_grid(theta, image.size(), align_corners=True).float()
            return F.grid_sample(image, grid, align_corners=True), theta
        if aug_name == 'JC':
            return mask_joint(image, preds * 4, self.cfg.MASK_JOINT_NUM), theta_identity
        if aug_name == 'JO':
            return keypoint_cutmix_perception(
                image, preds * 4, self.cfg.KP_CutMix_JOINT_NUM
            ), theta_identity
        if aug_name == 'CO':
            return random_cutout(image, self.cfg.MASK_HOLES_NUM_CO), theta_identity
        if aug_name == 'MU':
            return random_mixup(image), theta_identity
        if aug_name == 'CM':
            return random_cutmix(image, self.cfg.MASK_HOLES_NUM_CM), theta_identity
        if aug_name == 'DFBRA':
            return foreground_background_recomposition_aug(image, meta, self.cfg), theta_identity
        return image, theta_identity

    def _build_hard_views(self, base_image, randaug_image, student_aug, preds, meta=None):
        """Build hard views and record each view's geometric theta.

        Most hard augmentations are photometric/occlusion-like and keep identity
        geometry.  Geometric hard augmentations such as A60 still return theta so
        pseudo labels can be aligned consistently.  The extra UNSUP_TRANSFORM is
        still applied later exactly as before.
        """
        batch_identity = _identity_theta(base_image.shape[0], base_image.device)

        if (not self.cfg.MULTI_AUGS) and (not self.cfg.REPEAT_AUG):
            hard = randaug_image.clone() if ("RA" in student_aug or self.cfg.USE_RandAug_AUG) else base_image.clone()
            with torch.no_grad():
                hard, hard_theta = self._apply_student_flags_with_theta(hard, student_aug, preds, meta)
            return [hard], [hard_theta]

        hard_views = []
        hard_thetas = []
        repeat_times = self.cfg.REPEAT_TIMES if self.cfg.REPEAT_AUG else 1
        for _ in range(repeat_times):
            if self.cfg.USE_JointCutout_AUG:
                hard_views.append(mask_joint(base_image.clone(), preds * 4, self.cfg.MASK_JOINT_NUM))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_JointCutMix_AUG:
                hard_views.append(
                    keypoint_cutmix_perception(
                        base_image.clone(), preds * 4, self.cfg.KP_CutMix_JOINT_NUM
                    )
                )
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_Cutout_AUG:
                hard_views.append(random_cutout(base_image.clone(), self.cfg.MASK_HOLES_NUM_CO))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_MixUp_AUG:
                hard_views.append(random_mixup(base_image.clone()))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_CutMix_AUG:
                hard_views.append(random_cutmix(base_image.clone(), self.cfg.MASK_HOLES_NUM_CM))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_DiverRecomp_AUG:
                hard_views.append(foreground_background_recomposition_aug(base_image.clone(), meta, self.cfg))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_JointCutMix_CutOut_AUG:
                hard = keypoint_cutmix_perception(
                    base_image.clone(), preds * 4, self.cfg.KP_CutMix_JOINT_NUM
                )
                hard_views.append(random_cutout(hard, self.cfg.MASK_HOLES_NUM_CO))
                hard_thetas.append(batch_identity.clone())
            if self.cfg.USE_JointCutout_CutMix_AUG:
                hard = mask_joint(base_image.clone(), preds * 4, self.cfg.MASK_JOINT_NUM)
                hard_views.append(random_cutmix(hard, self.cfg.MASK_HOLES_NUM_CM))
                hard_thetas.append(batch_identity.clone())

        if not hard_views:
            hard_views.append(base_image.clone())
            hard_thetas.append(batch_identity.clone())
        return hard_views, hard_thetas

    def _apply_student_flags(self, image, student_aug, preds, meta=None):
        hard, _ = self._apply_student_flags_with_theta(image, student_aug, preds, meta)
        return hard

    def _apply_student_flags_with_theta(self, image, student_aug, preds, meta=None):
        """Apply the selected single-path hard augmentation and return geometry theta.

        Important ordering rule:
        DFBRA uses the current input-image coordinate system and its mask from
        meta['affine_trans'], so it must be applied before any geometric hard
        transform such as A60.  Otherwise the mask would be overlaid on an
        already affine-transformed image.
        """
        hard = image
        hard_theta = _identity_theta(image.shape[0], image.device)
        student_aug = str(student_aug)

        # Image-space / occlusion augmentations. They do not change coordinates.
        if "JC" in student_aug or self.cfg.USE_JointCutout_AUG:
            hard = mask_joint(hard, preds * 4, self.cfg.MASK_JOINT_NUM)
        if "JO" in student_aug or self.cfg.USE_JointCutMix_AUG:
            hard = keypoint_cutmix_perception(
                hard, preds * 4, self.cfg.KP_CutMix_JOINT_NUM
            )
        if "CO" in student_aug or self.cfg.USE_Cutout_AUG:
            hard = random_cutout(hard, self.cfg.MASK_HOLES_NUM_CO)
        if "MU" in student_aug or self.cfg.USE_MixUp_AUG:
            hard = random_mixup(hard)
        if "CM" in student_aug or self.cfg.USE_CutMix_AUG:
            hard = random_cutmix(hard, self.cfg.MASK_HOLES_NUM_CM)
        if "DFBRA" in student_aug or self.cfg.USE_DiverRecomp_AUG:
            hard = foreground_background_recomposition_aug(hard, meta, self.cfg)

        # Keep support for the original non-multi default/progressive A60 branch.
        # A60 is geometric, so it must update hard_theta.
        if "A60" in student_aug:
            a60_theta = self.get_batch_affine_transform(image.shape[0]).to(image.device)
            grid = F.affine_grid(a60_theta, hard.size(), align_corners=True).float()
            hard = F.grid_sample(hard, grid, align_corners=True)
            hard_theta = _compose_affine_theta(hard_theta, a60_theta)

        return hard, hard_theta

    def _warp_hard_and_teachers(self, hard_views, hard_view_thetas, kp_teacher, limb_teacher, teacher_theta):
        batch_size = kp_teacher.size(0)
        device = kp_teacher.device
        if hard_view_thetas is None:
            hard_view_thetas = [_identity_theta(batch_size, device) for _ in hard_views]
        hard_view_thetas = [theta.to(device=device, dtype=torch.float32) for theta in hard_view_thetas]

        if self.cfg.UNSUP_TRANSFORM:
            with torch.no_grad():
                extra_hard_theta = self.get_batch_affine_transform(batch_size).to(device)
                hard_grid = F.affine_grid(extra_hard_theta, hard_views[0].size(), align_corners=True).float()
                hard_views = [
                    F.grid_sample(hard, hard_grid, align_corners=True) for hard in hard_views
                ]
                final_hard_thetas = [
                    _compose_affine_theta(view_theta, extra_hard_theta)
                    for view_theta in hard_view_thetas
                ]
        else:
            extra_hard_theta = _identity_theta(batch_size, device)
            final_hard_thetas = hard_view_thetas

        kp_teacher_hard_list = []
        limb_teacher_hard_list = []
        teacher_to_hard_theta_list = []
        for final_theta in final_hard_thetas:
            teacher_to_hard_theta = _compose_teacher_to_hard_theta(
                teacher_theta.to(device), final_theta.to(device)
            )
            kp_grid = F.affine_grid(teacher_to_hard_theta, kp_teacher.size(), align_corners=True).float()
            limb_grid = F.affine_grid(teacher_to_hard_theta, limb_teacher.size(), align_corners=True).float()
            kp_teacher_hard_list.append(
                F.grid_sample(kp_teacher.detach(), kp_grid, align_corners=True)
            )
            limb_teacher_hard_list.append(
                F.grid_sample(limb_teacher.detach(), limb_grid, align_corners=True)
            )
            teacher_to_hard_theta_list.append(teacher_to_hard_theta)

        # Keep the old keys compatible by returning the first teacher tensor, while
        # still exposing per-hard-view teacher lists for multi-view hard augmentations.
        kp_teacher_hard = kp_teacher_hard_list[0]
        limb_teacher_hard = limb_teacher_hard_list[0]
        teacher_to_hard_theta = teacher_to_hard_theta_list[0]

        return (
            hard_views,
            kp_teacher_hard,
            limb_teacher_hard,
            kp_teacher_hard_list,
            limb_teacher_hard_list,
            extra_hard_theta,
            teacher_to_hard_theta,
            teacher_to_hard_theta_list,
        )

    def forward(self, x, meta=None, aug_couple=("A30", "A60")):
        """Forward pass following the original MultiAugs train/eval convention.

        The official MultiAugs training loop calls model.train() once before the
        training iterations, and validation calls model.eval().  Inside the
        model forward, the teacher pseudo-label branch uses torch.no_grad(), but
        it does not switch sub-networks between train/eval modes.  Therefore
        BatchNorm behavior is controlled only by the outer model.train()/eval()
        call, exactly as in the official pose_dual implementation.
        """
        if not isinstance(x, list):
            return _heatmap(self.keypoint_net(x))

        teacher_aug, student_aug = aug_couple
        unsup_meta = None
        if isinstance(meta, (list, tuple)) and len(meta) > 1:
            unsup_meta = meta[1]
        elif isinstance(meta, dict):
            unsup_meta = meta

        if self.cfg.USE_RandAug_AUG:
            sup_x, unsup_x, aug_unsup_x = x
        else:
            sup_x, unsup_x = x
            aug_unsup_x = unsup_x

        # Official MultiAugs order: supervised branch first, using the current
        # outer mode. During training this is train-mode BatchNorm; during
        # validation/inference it is eval-mode BatchNorm.
        sup_kp = _heatmap(self.keypoint_net(sup_x))
        sup_limb = _heatmap(self.limb_net(sup_x))

        # Teacher and augmentation construction follow the official pattern:
        # no gradient is recorded, but train/eval mode is NOT changed here.
        with torch.no_grad():
            easy_x = unsup_x.clone()
            if teacher_aug in ['JC', 'JO']:
                kp_for_aug = _heatmap(self.keypoint_net(easy_x))
                preds, _ = get_max_preds_tensor(kp_for_aug.detach())
                easy_x, teacher_theta = self._apply_named_aug_with_theta(
                    easy_x, teacher_aug, preds, unsup_meta
                )
            else:
                easy_x, teacher_theta = self._apply_named_aug_with_theta(
                    easy_x, teacher_aug, meta=unsup_meta
                )

            unsup_kp_easy = _heatmap(self.keypoint_net(easy_x))
            unsup_limb_easy = _heatmap(self.limb_net(easy_x))
            preds, _ = get_max_preds_tensor(unsup_kp_easy.detach())

            hard_views, hard_view_thetas = self._build_hard_views(
                unsup_x.clone(), aug_unsup_x.clone(), student_aug, preds, unsup_meta
            )
            (
                hard_views,
                kp_teacher_hard,
                limb_teacher_hard,
                kp_teacher_hard_list,
                limb_teacher_hard_list,
                theta,
                teacher_to_hard_theta,
                teacher_to_hard_theta_list,
            ) = self._warp_hard_and_teachers(
                hard_views,
                hard_view_thetas,
                unsup_kp_easy,
                unsup_limb_easy,
                teacher_theta,
            )

        # Student hard branch also follows the official pattern: it is computed
        # in the current outer mode and keeps gradients. No internal BN freezing
        # or eval switching is applied here.
        hard_kp_list = [_heatmap(self.keypoint_net(hard)) for hard in hard_views]
        hard_limb_list = [_heatmap(self.limb_net(hard)) for hard in hard_views]

        return {
            'sup_kp': sup_kp,
            'sup_limb': sup_limb,
            'unsup_kp_easy': unsup_kp_easy,
            'unsup_limb_easy': unsup_limb_easy,
            'kp_teacher_hard': kp_teacher_hard,
            'limb_teacher_hard': limb_teacher_hard,
            'kp_teacher_hard_list': kp_teacher_hard_list,
            'limb_teacher_hard_list': limb_teacher_hard_list,
            'hard_kp_list': hard_kp_list,
            'hard_limb_list': hard_limb_list,
            'out_dic': {
                'easy_x': easy_x,
                'hard_views': hard_views,
                'theta': theta,
                'teacher_to_hard_theta': teacher_to_hard_theta,
                'teacher_to_hard_theta_list': teacher_to_hard_theta_list,
            },
        }


def get_pose_net(cfg, is_train, **kwargs):
    keypoint_cfg = _make_hrnet_cfg(cfg, cfg.MODEL)
    limb_cfg = _make_hrnet_cfg(cfg, cfg.LIMB_MODEL)

    keypoint_net = PoseHighResolutionNet(keypoint_cfg, prm=getattr(cfg.MODEL, 'PRM', False))
    limb_net = PoseHighResolutionNet(limb_cfg, prm=getattr(cfg.LIMB_MODEL, 'PRM', False))

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        keypoint_net.init_weights(cfg.MODEL.PRETRAINED)
    if is_train and cfg.LIMB_MODEL.INIT_WEIGHTS:
        limb_net.init_weights(cfg.LIMB_MODEL.PRETRAINED)

    logger.info(
        'Hetero limb model: keypoint channels={}, limb channels={}'.format(
            cfg.MODEL.NUM_JOINTS, cfg.LIMB_MODEL.NUM_JOINTS
        )
    )
    return HeteroLimbCotrain(keypoint_net, limb_net, cfg)
