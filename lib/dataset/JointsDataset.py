# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.limb_heatmap import build_limb_pairs
from utils.limb_heatmap import get_mpii_parent_ids
from utils.limb_heatmap import generate_limb_target


logger = logging.getLogger(__name__)


SKELETON = [
    [6,2],[1,2],[1,0],[6,3],[4,3],[5,4],[6,7],[7,8],[8,9],[8,12],[12,11],[11,10],[8,13],[13,14],[14,15]
]
CocoColors = [[0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255],
              [0, 0, 255], [0, 0, 255], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255],
              [0, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
NUM_KPTS = 16


def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1]
        if x_a > 0 and y_a > 0 and x_b > 0 and y_b > 0:
            cv2.circle(img, (int(x_a), int(y_a)), 3, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 3, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    cv2.circle(img, (int(keypoints[6][0]), int(keypoints[6][1])), 3, (255,255,255), -1)

def bbox_to_xyxy_array(bbox):
    """Convert a bbox polygon/list of polygons to a stable [x1, y1, x2, y2] array.

    The DFBRA mask library stores instance boxes in xyxy format.  Storing the
    meta bbox in the same format also avoids DataLoader collation failures when
    a dataset record contains a list of polygons with different lengths.
    """
    pts_list = []
    if isinstance(bbox, list):
        for b in bbox:
            arr = np.asarray(b, dtype=np.float32)
            if arr.size >= 4:
                pts_list.append(arr.reshape(-1, 2))
    else:
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.size >= 4:
            pts_list.append(arr.reshape(-1, 2))

    if not pts_list:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    pts = np.concatenate(pts_list, axis=0)
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def encode_path_to_fixed_uint8(path_like, max_len=2048):
    """Encode an image path as a fixed-length uint8 array.

    Why this is needed:
        In torch.nn.DataParallel, tensors are split along the batch dimension,
        but Python lists of strings in meta['image'] are replicated to every
        GPU.  DFBRA needs the local image path to match local tensor fields
        such as bbox_orig and affine_trans.  Storing the path additionally as
        a tensor lets DataParallel split it correctly.
    """
    arr = np.zeros((int(max_len),), dtype=np.uint8)
    encoded = str(path_like).encode('utf-8', errors='ignore')[:int(max_len) - 1]
    if len(encoded) > 0:
        arr[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return arr


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = get_mpii_parent_ids()

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.enable_limb_targets = bool(
            getattr(getattr(cfg, 'HETERO_LIMB', {}), 'ENABLED', False)
            or cfg.MODEL.NAME == 'pose_hetero_limb'
        )
        self.limb_pairs = build_limb_pairs(self.parent_ids, 16)
        self.limb_heatmap_sigma = float(
            getattr(cfg.DATASET, 'LIMB_HEATMAP_SIGMA', getattr(getattr(cfg, 'HETERO_LIMB', {}), 'LIMB_HEATMAP_SIGMA', 2.0))
        )
        self.limb_heatmap_truncate = float(
            getattr(cfg.DATASET, 'LIMB_HEATMAP_TRUNCATE', getattr(getattr(cfg, 'HETERO_LIMB', {}), 'LIMB_HEATMAP_TRUNCATE', 3.0))
        )

        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.transform = transform
        self.db = []

        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height

        # The original MultiAugs code uses a 24-joint union space to mix COCO/MPII/AIC.
        # For MPII-style experiments, set MODEL.NUM_JOINTS: 16 in the yaml file.
        # Then all datasets are mapped into the MPII 16-joint order, so the HRNet head
        # and the supervised HRNet baseline output the same number/order of heatmaps.
        self.num_joints = int(cfg.MODEL.NUM_JOINTS)
        if self.num_joints == 16:
            # MPII original 16-joint order:
            # 0 rank, 1 rkne, 2 rhip, 3 lhip, 4 lkne, 5 lank,
            # 6 root, 7 spine, 8 thorax, 9 head,
            # 10 rwri, 11 relb, 12 rsho, 13 lsho, 14 lelb, 15 lwri
            self.union_joints = {
                0: 'rank',
                1: 'rkne',
                2: 'rhip',
                3: 'lhip',
                4: 'lkne',
                5: 'lank',
                6: 'root',
                7: 'spine',
                8: 'thorax',
                9: 'head',
                10: 'rwri',
                11: 'relb',
                12: 'rsho',
                13: 'lsho',
                14: 'lelb',
                15: 'lwri',
            }
            self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15)
            self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
            # Frogman-16 uses 7=spine, 8=thorax, 9=head.
            # AIC labels are dummy/unlabeled here; these aliases only keep channel masks usable.
            self.joint_name_aliases = {'neck': 'thorax', 'head top': 'head'}
        elif self.num_joints == 24:
            self.union_joints = {
                0: 'root',
                1: 'rhip',
                2: 'rkne',
                3: 'rank',
                4: 'lhip',
                5: 'lkne',
                6: 'lank',
                7: 'belly',
                8: 'thorax',
                9: 'neck',
                10: 'upper neck',
                11: 'nose',
                12: 'head',
                13: 'head top',
                14: 'lsho',
                15: 'lelb',
                16: 'lwri',
                17: 'rsho',
                18: 'relb',
                19: 'rwri',
                20: 'reye',
                21: 'leye',
                22: 'rear',
                23: 'lear',
            }
            self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
            self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
            self.joint_name_aliases = {}
        else:
            raise ValueError('This project currently supports MODEL.NUM_JOINTS = 16 or 24, got {}'.format(self.num_joints))

        self.actual_joints = {}
        self.u2a_mapping = {}


    def set_limb_pairs_from_parent_ids(self, parent_ids):
        if self.enable_limb_targets and self.num_joints != 16:
            raise ValueError(
                'Heterogeneous limb training expects MPII-style 16 joints, got {}'.format(
                    self.num_joints
                )
            )
        self.parent_ids = list(parent_ids)
        self.limb_pairs = build_limb_pairs(self.parent_ids, self.num_joints)


    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            # Some datasets contain a semantically equivalent joint under a different
            # name. Example: AIC uses "neck" while MPII-16 uses "upper neck".
            # Map the alias before checking the selected output space.
            mapped_name = self.joint_name_aliases.get(v, v)
            if mapped_name not in union_values:
                continue
            idx = union_values.index(mapped_name)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_mapping(self):
        mapping = self.u2a_mapping
        for item in self.db:
            joints = item['joints_3d']
            joints_vis = item['joints_3d_vis']

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 3), dtype=np.float32)
            joints_union_vis = np.zeros(shape=(njoints, 3), dtype=np.float32)

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_3d'] = joints_union
            item['joints_3d_vis'] = joints_union_vis


    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx, rot_angle=None, scale=None, flip=None):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        #bbox
        bbox = np.zeros((4,2)).astype(np.int32)
        if 'bbox' in db_rec.keys():
            bbox = db_rec['bbox']
        bbox_orig = bbox_to_xyxy_array(bbox)
        orig_image_shape = np.array(data_numpy.shape[:2], dtype=np.int32)  # [height, width]

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # draw_pose(joints[:, 0:2], data_numpy)
        # cv2.imshow('h', data_numpy)
        # cv2.waitKey()

        if self.is_train:
            # Exclude the unlabeled
            if 'unlabel' not in self.image_set:
                if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                    c_half_body, s_half_body = self.half_body_transform(
                        joints, joints_vis
                    )

                    if c_half_body is not None and s_half_body is not None:
                        c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor

            if scale is not None:
                s = s * scale
            else:
                scale = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
                s = s * scale
                
            if rot_angle is not None:
                r = rot_angle
            else:
                r = np.clip(np.random.randn()*rf, -rf, rf) \
                    if random.random() <= 0.6 else 0
            if flip is not None:
                f = flip
            else:
                f = random.random() <= 0.5

            if self.flip and f:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

                if type(bbox) == list:
                    for b in bbox:
                        b[:, 0] = data_numpy.shape[1] - b[:, 0] - 1  
                else:
                    bbox[:, 0] = data_numpy.shape[1] - bbox[:, 0] - 1
        else:
            scale = 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        # input2 = input.copy()
        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # bbox
        if type(bbox) == list:
            for b in bbox:
                for i in range(len(b)):
                    b[i] = affine_transform(b[i], trans)      
                b = b.astype(np.int32)
        else:
            for i in range(4):
                bbox[i] = affine_transform(bbox[i], trans)      
            bbox = bbox.astype(np.int32)
        # cv2.namedWindow('hello', cv2.WINDOW_NORMAL)
        # draw_pose(joints[:, 0:2], input2)
        # cv2.imshow('hello', input2)
        # cv2.waitKey()

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        mask_target = self.generate_mask(bbox)
        mask_target = torch.from_numpy(mask_target)

        # Just to Fill the Field
        if 'raw_box' in db_rec:
            raw_box = db_rec['raw_box'].astype(float)
        else:
            raw_box = np.array([1,1,2,2]).astype(float)

        # s = s * scale
        meta = {
            'image': image_file,
            # Tensor-encoded path for DataParallel-safe DFBRA.  meta['image']
            # remains for logging/debugging, while image_path_bytes is split
            # correctly together with tensor fields on non-distributed multi-GPU.
            'image_path_bytes': encode_path_to_fixed_uint8(image_file),
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'scale_ratio': scale,
            'rotation': r,
            'affine_trans': trans.astype(np.float32),
            'bbox_orig': bbox_orig.astype(np.float32),
            'orig_image_shape': orig_image_shape,
            'score': score,
            'raw_box': raw_box, 
            'mask_target':mask_target,
        }

        if self.enable_limb_targets:
            limb_target, limb_target_weight = generate_limb_target(
                joints,
                joints_vis,
                self.image_size,
                self.heatmap_size,
                self.limb_pairs,
                sigma=self.limb_heatmap_sigma,
                truncate=self.limb_heatmap_truncate,
            )
            meta['limb_target'] = torch.from_numpy(limb_target)
            meta['limb_target_weight'] = torch.from_numpy(limb_target_weight)

        return input, target, target_weight, meta


    def generate_mask(self, bbox):
        # feat_stride = self.image_size / self.heatmap_size
        # bbox = bbox.copy()
        # bbox[:,0] = (bbox[:,0] / feat_stride[0])
        # bbox[:,1] = (bbox[:,1] / feat_stride[1])


        # x1,y1=np.min(bbox,0)
        # x2,y2=np.max(bbox,0)
        # bbox = np.array([[x1, y1],[x1, y2],[x2, y2],[x2, y1]]).astype(np.int32)

        target = np.zeros((
                    self.image_size[1],
                    self.image_size[0]),
                    dtype=np.float32)

        if type(bbox) == list:
            fore_mask = cv2.fillPoly(target.copy(), bbox, 1)
        else:
            fore_mask = cv2.fillPoly(target.copy(), [bbox], 1)
        back_mask = 1 - fore_mask

        target = np.stack([fore_mask,back_mask],0)

        # target = np.zeros((2,
        #             self.image_size[1],
        #             self.image_size[0]),
        #             dtype=np.float32)
        # # print(bbox.dtype)
        # # foreground
        # target[0,bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]] = 1
        # # background
        # target[1] = 1
        # target[1,bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]] = 0

        return target


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
