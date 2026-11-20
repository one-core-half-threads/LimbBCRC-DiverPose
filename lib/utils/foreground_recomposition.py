# -*- coding: utf-8 -*-
"""
Diver foreground-background recomposition augmentation (DFBRA).

This module is intentionally independent from the GrabCut implementation.  The
GrabCut tool writes an instance-level mask library (mask_index.json + mask pngs),
and this module reads that library during semi-supervised training.

Input image tensors are expected to be ImageNet-normalized RGB tensors with
shape [N, 3, H, W], which is consistent with the HRNet/MultiAugs preprocessing.
The augmentation keeps the diver foreground at the same position in the current
network input; it only replaces the background with a crop sampled from another
unlabeled underwater image outside its annotated diver boxes. Therefore pseudo
keypoints/heatmaps remain valid and do not need geometric modification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

_AUGMENTER_CACHE = {}


def _cfg_get(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _resolve_path(path_like: str, cfg=None) -> Path:
    p = Path(str(path_like))
    if p.is_absolute():
        return p
    # Prefer cwd-relative path because training commands are usually launched
    # from the project root.  Fall back to DATASET.ROOT-relative path.
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return cwd_p
    data_root = getattr(getattr(cfg, 'DATASET', None), 'ROOT', '') if cfg is not None else ''
    if data_root:
        root_p = Path(str(data_root)) / p
        if root_p.exists():
            return root_p
    return cwd_p


def _read_image_rgb(path: str, color_rgb: bool = True) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except Exception:
        img = None
    if img is None:
        return None
    if color_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _read_mask(path: str) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        mask = None
    return mask


def _xyxy_from_poly(poly) -> Optional[Tuple[float, float, float, float]]:
    arr = np.asarray(poly, dtype=np.float32)
    if arr.size == 4 and arr.ndim == 1:
        x1, y1, x2, y2 = arr.tolist()
        return float(x1), float(y1), float(x2), float(y2)
    arr = arr.reshape(-1, 2)
    if arr.shape[0] < 2:
        return None
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _meta_value(meta: Dict[str, Any], key: str, index: int, default=None):
    """Safely read a meta field for a local sample.

    Most training calls use DataLoader-collated metadata, e.g.
    bbox_orig -> [B, 4], affine_trans -> [B, 2, 3],
    image_path_bytes -> [B, L].  Some debugging calls may pass a single
    uncollated dataset item instead, e.g. bbox_orig -> [4],
    affine_trans -> [2, 3], image_path_bytes -> [L].

    The original generic indexing rule works for collated batches, but it
    incorrectly turns an uncollated fixed-length path array into only its
    first byte and turns an uncollated bbox/affine matrix into only its first
    row.  Handle those shapes explicitly so DFBRA also works in one-sample
    debug/inference checks.
    """
    if meta is None or key not in meta:
        return default

    value = meta[key]

    if isinstance(value, torch.Tensor):
        if key == 'image_path_bytes':
            item = value if value.dim() == 1 else value[index]
            return item.detach().cpu().numpy()
        if key == 'bbox_orig':
            item = value if value.dim() == 1 else value[index]
            if item.numel() == 1:
                return item.item()
            return item.detach().cpu().numpy()
        if key == 'affine_trans':
            item = value if value.dim() == 2 else value[index]
            if item.numel() == 1:
                return item.item()
            return item.detach().cpu().numpy()

        item = value[index]
        if item.numel() == 1:
            return item.item()
        return item.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if key == 'image_path_bytes':
            return value if value.ndim == 1 else value[index]
        if key == 'bbox_orig':
            return value if value.ndim == 1 else value[index]
        if key == 'affine_trans':
            return value if value.ndim == 2 else value[index]
        return value[index]

    if isinstance(value, (list, tuple)):
        return value[index]

    return value


def _decode_uint8_path_array(x) -> str:
    """Decode a zero-padded uint8 array/tensor back to a UTF-8 path."""
    if x is None:
        return ''
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    try:
        arr = arr.astype(np.uint8).reshape(-1)
    except Exception:
        return ''
    if arr.size == 0:
        return ''
    nz = np.where(arr == 0)[0]
    end = int(nz[0]) if nz.size > 0 else int(arr.size)
    if end <= 0:
        return ''
    try:
        return bytes(arr[:end].tolist()).decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _meta_image_path(meta: Dict[str, Any], index: int, default='') -> str:
    """Return the image path for local sample index.

    In DataParallel, meta['image'] is a Python list of strings and may be
    replicated to every GPU instead of being split with the tensor batch.
    JointsDataset additionally stores meta['image_path_bytes'] as a tensor,
    which DataParallel does split correctly.  Prefer that local tensor value
    when available, and fall back to meta['image'] for single-GPU/DDP/debug.
    """
    encoded = _meta_value(meta, 'image_path_bytes', index, None)
    path = _decode_uint8_path_array(encoded)
    if path:
        return path
    return str(_meta_value(meta, 'image', index, default))


def _to_bool(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.item()) if x.numel() == 1 else bool(x.detach().cpu().numpy().reshape(-1)[0])
    if isinstance(x, np.ndarray):
        return bool(x.reshape(-1)[0]) if x.size > 0 else False
    return bool(x)


def _as_np_float_array(x, shape=None) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if shape is not None:
        try:
            arr = arr.reshape(shape)
        except Exception:
            return None
    return arr


def _crop_intersects_boxes(crop_xyxy, boxes_xyxy) -> bool:
    x1, y1, x2, y2 = crop_xyxy
    for b in boxes_xyxy:
        if _box_iou((x1, y1, x2, y2), b) > 0.0:
            return True
    return False


def _rgb_crop_to_normalized_tensor(crop_rgb: np.ndarray, device, dtype) -> torch.Tensor:
    crop = crop_rgb.astype(np.float32) / 255.0
    crop = crop.transpose(2, 0, 1)
    crop = (crop - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(crop).to(device=device, dtype=dtype)


class DiverForegroundBackgroundRecomposer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        # This augmenter is only called from the DFBRA branch.  Therefore, once
        # called, it should be allowed to run even if the branch was selected by
        # PROG_MODE rather than the global USE_DiverRecomp_AUG flag.  Missing
        # index files still make it a safe no-op.
        self.enabled = True
        self.mask_index_path = _resolve_path(_cfg_get(cfg, 'DFBRA_MASK_INDEX', ''), cfg)
        self.ratio_min = float(_cfg_get(cfg, 'DFBRA_RATIO_MIN', 0.28))
        self.ratio_max = float(_cfg_get(cfg, 'DFBRA_RATIO_MAX', 0.85))
        self.prob = float(_cfg_get(cfg, 'DFBRA_PROB', 1.0))
        self.max_bg_image_tries = int(_cfg_get(cfg, 'DFBRA_MAX_BG_IMAGE_TRIES', 30))
        self.max_bg_crop_tries = int(_cfg_get(cfg, 'DFBRA_MAX_BG_CROP_TRIES', 50))
        self.feather_kernel = int(_cfg_get(cfg, 'DFBRA_FEATHER_KERNEL', 7))
        self.min_iou_match = float(_cfg_get(cfg, 'DFBRA_MIN_BBOX_IOU', 0.30))
        self.max_mask_cache = int(_cfg_get(cfg, 'DFBRA_MAX_MASK_CACHE', 512))
        self.max_image_cache = int(_cfg_get(cfg, 'DFBRA_MAX_IMAGE_CACHE', 128))
        self.color_rgb = bool(getattr(getattr(cfg, 'DATASET', None), 'COLOR_RGB', True))

        self.records: List[Dict[str, Any]] = []
        self.instances_by_stem: Dict[str, List[Dict[str, Any]]] = {}
        self.records_by_stem: Dict[str, Dict[str, Any]] = {}
        self.all_records: List[Dict[str, Any]] = []
        self.loaded = False
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._image_cache: Dict[str, np.ndarray] = {}

        if self.enabled:
            self._load_index()

    def _load_index(self):
        if self.loaded:
            return
        if not self.mask_index_path or not self.mask_index_path.exists():
            self.loaded = True
            return

        with open(self.mask_index_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        if isinstance(payload, dict) and 'records' in payload:
            records = payload.get('records', [])
            mask_dir = payload.get('mask_dir', str(self.mask_index_path.parent / 'instance_masks'))
        elif isinstance(payload, list):
            records = payload
            mask_dir = str(self.mask_index_path.parent / 'instance_masks')
        else:
            records = []
            mask_dir = str(self.mask_index_path.parent / 'instance_masks')

        mask_dir = _resolve_path(mask_dir, self.cfg)
        for rec in records:
            stem = str(rec.get('stem', Path(str(rec.get('image_name', ''))).stem))
            if not stem:
                continue
            image_path_raw = rec.get('image', '')
            image_path = str(_resolve_path(image_path_raw, self.cfg)) if image_path_raw else ''
            instances = rec.get('instances', [])
            valid_instances = []
            all_boxes = []
            for inst in instances:
                raw_box = inst.get('raw_bbox_xyxy', None)
                if raw_box is None:
                    continue
                raw_box = tuple(map(float, raw_box))
                all_boxes.append(raw_box)
                ratio = float(inst.get('mask_rect_ratio', -1.0))
                if ratio < self.ratio_min or ratio > self.ratio_max:
                    continue
                rel = inst.get('mask_rel_path', '')
                mask_path = Path(str(inst.get('mask_path', '')))
                if not mask_path.exists():
                    mask_path = mask_dir / rel if rel else mask_path
                inst2 = dict(inst)
                inst2['mask_path'] = str(mask_path)
                inst2['image'] = image_path
                inst2['stem'] = stem
                inst2['raw_bbox_xyxy_tuple'] = raw_box
                valid_instances.append(inst2)
            rec2 = dict(rec)
            rec2['image'] = image_path
            rec2['all_boxes_xyxy'] = all_boxes
            rec2['valid_instances'] = valid_instances
            self.records_by_stem[stem] = rec2
            self.instances_by_stem[stem] = valid_instances
            self.all_records.append(rec2)

        self.loaded = True

    def _get_matching_instance(self, image_path: str, bbox_poly) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        self._load_index()
        stem = Path(str(image_path)).stem
        candidates = self.instances_by_stem.get(stem, [])
        if not candidates:
            return None
        target_box = _xyxy_from_poly(bbox_poly)
        if target_box is None:
            return None
        best = None
        best_iou = 0.0
        for inst in candidates:
            iou = _box_iou(target_box, inst['raw_bbox_xyxy_tuple'])
            if iou > best_iou:
                best_iou = iou
                best = inst
        if best is None or best_iou < self.min_iou_match:
            return None
        return best

    def _load_cached_mask(self, mask_path: str) -> Optional[np.ndarray]:
        if mask_path in self._mask_cache:
            return self._mask_cache[mask_path]
        mask = _read_mask(mask_path)
        if mask is not None:
            if self.max_mask_cache > 0 and len(self._mask_cache) >= self.max_mask_cache:
                self._mask_cache.clear()
            self._mask_cache[mask_path] = mask
        return mask

    def _load_cached_image(self, image_path: str) -> Optional[np.ndarray]:
        if image_path in self._image_cache:
            return self._image_cache[image_path]
        img = _read_image_rgb(image_path, self.color_rgb)
        if img is not None:
            # Keep cache small enough for long training; random access is optional.
            if self.max_image_cache > 0 and len(self._image_cache) >= self.max_image_cache:
                self._image_cache.clear()
            self._image_cache[image_path] = img
        return img

    def _warp_mask_to_input(self, mask: np.ndarray, meta: Dict[str, Any], index: int, out_h: int, out_w: int) -> Optional[np.ndarray]:
        trans = _as_np_float_array(_meta_value(meta, 'affine_trans', index, None), (2, 3))
        if trans is None:
            return None
        warped = cv2.warpAffine(
            mask,
            trans,
            (int(out_w), int(out_h)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped = (warped > 0).astype(np.uint8) * 255
        if cv2.countNonZero(warped) <= 0:
            return None
        return warped

    def _sample_background_crop(self, out_h: int, out_w: int, exclude_stem: str = '') -> Optional[np.ndarray]:
        if not self.all_records:
            return None
        for _ in range(self.max_bg_image_tries):
            rec = random.choice(self.all_records)
            stem = str(rec.get('stem', ''))
            if len(self.all_records) > 1 and stem == exclude_stem:
                continue
            image_path = str(rec.get('image', ''))
            img = self._load_cached_image(image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < out_h or w < out_w:
                continue
            boxes = rec.get('all_boxes_xyxy', [])
            for _ in range(self.max_bg_crop_tries):
                x1 = random.randint(0, w - out_w)
                y1 = random.randint(0, h - out_h)
                crop_box = (x1, y1, x1 + out_w, y1 + out_h)
                if _crop_intersects_boxes(crop_box, boxes):
                    continue
                crop = img[y1:y1 + out_h, x1:x1 + out_w].copy()
                if crop.shape[0] == out_h and crop.shape[1] == out_w:
                    return crop
        return None

    def __call__(self, images: torch.Tensor, meta: Optional[Dict[str, Any]]) -> torch.Tensor:
        if (not self.enabled) or meta is None or images is None or images.dim() != 4:
            return images
        if random.random() > self.prob:
            return images

        out = images.clone()
        n, c, h, w = out.shape
        if c != 3:
            return out

        for i in range(n):
            image_path = _meta_image_path(meta, i, '')
            bbox_poly = _meta_value(meta, 'bbox_orig', i, None)
            if not image_path or bbox_poly is None:
                continue

            inst = self._get_matching_instance(str(image_path), bbox_poly)
            if inst is None:
                continue
            mask = self._load_cached_mask(str(inst.get('mask_path', '')))
            if mask is None:
                continue
            mask_input = self._warp_mask_to_input(mask, meta, i, int(h), int(w))
            if mask_input is None:
                continue

            bg_crop = self._sample_background_crop(int(h), int(w), exclude_stem=Path(str(image_path)).stem)
            if bg_crop is None:
                continue

            alpha = mask_input.astype(np.float32) / 255.0
            if self.feather_kernel > 1:
                k = int(self.feather_kernel)
                if k % 2 == 0:
                    k += 1
                alpha = cv2.GaussianBlur(alpha, (k, k), 0)
                alpha = np.clip(alpha, 0.0, 1.0)
            alpha_t = torch.from_numpy(alpha).to(device=out.device, dtype=out.dtype).view(1, int(h), int(w))
            bg_t = _rgb_crop_to_normalized_tensor(bg_crop, out.device, out.dtype)
            out[i] = out[i] * alpha_t + bg_t * (1.0 - alpha_t)

        return out


def foreground_background_recomposition_aug(images: torch.Tensor,
                                            meta: Optional[Dict[str, Any]],
                                            cfg) -> torch.Tensor:
    """Cached entry point used by pose models."""
    key = id(cfg)
    aug = _AUGMENTER_CACHE.get(key, None)
    if aug is None:
        aug = DiverForegroundBackgroundRecomposer(cfg)
        _AUGMENTER_CACHE[key] = aug
    return aug(images, meta)
