from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


MPII_PARENT_IDS = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 8, 8, 13, 14]


def get_mpii_parent_ids():
    """Return the canonical MPII-16 child->parent definition used by this project.

    The joint order is:
    0 rank, 1 rkne, 2 rhip, 3 lhip, 4 lkne, 5 lank,
    6 root, 7 spine, 8 thorax, 9 head,
    10 rwri, 11 relb, 12 rsho, 13 lsho, 14 lelb, 15 lwri.
    """
    return list(MPII_PARENT_IDS)


def build_limb_pairs(parent_ids, num_joints=16):
    pairs = []
    for child, parent in enumerate(parent_ids):
        parent = int(parent)
        if child == parent:
            continue
        if child < num_joints and parent < num_joints:
            pairs.append((child, parent))
    if not pairs:
        raise ValueError('No valid limb pairs can be built from parent_ids.')
    return pairs


MPII_LIMB_PAIRS = build_limb_pairs(get_mpii_parent_ids(), 16)


def get_limb_flip_pairs(limb_pairs=MPII_LIMB_PAIRS, joint_flip_pairs=None):
    if joint_flip_pairs is None:
        joint_flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

    joint_swap = {i: i for pair in limb_pairs for i in pair}
    for left, right in joint_flip_pairs:
        joint_swap[int(left)] = int(right)
        joint_swap[int(right)] = int(left)

    canonical_to_index = {frozenset(pair): idx for idx, pair in enumerate(limb_pairs)}
    limb_flip_pairs = []
    for idx, (child, parent) in enumerate(limb_pairs):
        flipped = frozenset((joint_swap[child], joint_swap[parent]))
        flipped_idx = canonical_to_index.get(flipped)
        if flipped_idx is not None and flipped_idx != idx and idx < flipped_idx:
            limb_flip_pairs.append((idx, flipped_idx))
    return limb_flip_pairs


def generate_limb_target(
    joints,
    joints_vis,
    image_size,
    heatmap_size,
    limb_pairs=MPII_LIMB_PAIRS,
    sigma=2.0,
    truncate=3.0,
):
    num_limbs = len(limb_pairs)
    heatmap_w, heatmap_h = int(heatmap_size[0]), int(heatmap_size[1])
    target = np.zeros((num_limbs, heatmap_h, heatmap_w), dtype=np.float32)
    target_weight = np.ones((num_limbs, 1), dtype=np.float32)

    feat_stride = np.asarray(image_size, dtype=np.float32) / np.asarray(
        heatmap_size, dtype=np.float32
    )
    sigma = max(float(sigma), 1e-6)
    truncate = max(float(truncate), 1.0)
    radius = sigma * truncate

    for limb_id, (joint_from, joint_to) in enumerate(limb_pairs):
        if joints_vis[joint_from, 0] <= 0 or joints_vis[joint_to, 0] <= 0:
            target_weight[limb_id, 0] = 0.0
            continue

        p1 = np.array(
            [
                joints[joint_from, 0] / feat_stride[0],
                joints[joint_from, 1] / feat_stride[1],
            ],
            dtype=np.float32,
        )
        p2 = np.array(
            [
                joints[joint_to, 0] / feat_stride[0],
                joints[joint_to, 1] / feat_stride[1],
            ],
            dtype=np.float32,
        )

        vec = p2 - p1
        length_sq = float(np.dot(vec, vec))
        if length_sq < 1e-6:
            target_weight[limb_id, 0] = 0.0
            continue

        x_min = int(max(0, np.floor(min(p1[0], p2[0]) - radius)))
        x_max = int(min(heatmap_w - 1, np.ceil(max(p1[0], p2[0]) + radius)))
        y_min = int(max(0, np.floor(min(p1[1], p2[1]) - radius)))
        y_max = int(min(heatmap_h - 1, np.ceil(max(p1[1], p2[1]) + radius)))
        if x_max < x_min or y_max < y_min:
            target_weight[limb_id, 0] = 0.0
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32)
        ys = np.arange(y_min, y_max + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x, grid_y], axis=-1)

        ap = points - p1.reshape(1, 1, 2)
        t = (ap[..., 0] * vec[0] + ap[..., 1] * vec[1]) / length_sq
        t = np.clip(t, 0.0, 1.0)
        closest = p1.reshape(1, 1, 2) + t[..., None] * vec.reshape(1, 1, 2)
        dist_sq = np.sum((points - closest) ** 2, axis=-1)
        tube = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
        tube[dist_sq > radius * radius] = 0.0

        target[limb_id, y_min:y_max + 1, x_min:x_max + 1] = np.maximum(
            target[limb_id, y_min:y_max + 1, x_min:x_max + 1],
            tube,
        )

    return target, target_weight
