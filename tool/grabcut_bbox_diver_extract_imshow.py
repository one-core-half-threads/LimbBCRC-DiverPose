# -*- coding: utf-8 -*-
"""
独立测试脚本：bbox + GrabCut + 连通域过滤 + 形态学后处理抠出蛙人前景

功能：
1. 遍历 IMAGE_DIR 下的图片。
2. 读取同名 LabelMe json 中的 rectangle / polygon 外接框作为蛙人 bbox。
3. 以 bbox 初始化 GrabCut，得到蛙人前景 mask。
4. 对 mask 做形态学闭运算、开运算、填洞、连通域过滤。
5. cv2.imshow 显示：原图+bbox、mask、抠图预览、背景替换预览。
6. 可选保存 mask、透明 PNG 抠图、预览图。

依赖：
    pip install opencv-python numpy

用法：
    1. 修改 CONFIG 区域的 IMAGE_DIR、LABELME_JSON_DIR、OUTPUT_DIR。
    2. 直接运行：
        python grabcut_bbox_diver_extract_imshow.py

按键：
    任意键：下一张
    q 或 ESC：退出
    s：保存当前图片结果，即使 SAVE_RESULTS=False
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG：优先修改这里
# =========================

IMAGE_DIR = r"data/DiversUnlabeled/train/images"
LABELME_JSON_DIR = r"data/DiversUnlabeled/train/images"
OUTPUT_DIR = r"data/DiversUnlabeled/train/grabcut_result"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_PROCESS_IMAGES = 0  # <=0 表示全部处理

# 是否保存结果。
SAVE_RESULTS = True
SAVE_MASK = True
SAVE_CUTOUT_RGBA = True
SAVE_PREVIEW = True

# LabelMe label 过滤。
# None 表示读取所有能形成 bbox 的 rectangle / polygon。
# 如果你的 json 里有很多非人体 shape，建议改成：
# BBOX_LABELS_TO_USE = ["person", "diver", "body"]
BBOX_LABELS_TO_USE: Optional[Sequence[str]] = None

# bbox 扩大比例。bbox 太紧时，适当扩大可以避免把蛙人边界误删。
BBOX_EXPAND_RATIO = 0.05

# GrabCut 迭代次数。一般 3~8。
GRABCUT_ITER_COUNT = 3

# 是否在 GrabCut 前对 ROI 做轻微去噪。
# 水下图像噪声很重时可打开；但过强平滑可能损伤细边界。
USE_BILATERAL_FILTER = False
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 40

# 形态学后处理核大小。
# CLOSE 用于连接断裂区域，OPEN 用于去除小噪声。
MORPH_CLOSE_KERNEL = 9
MORPH_CLOSE_ITER = 2
MORPH_OPEN_KERNEL = 5
MORPH_OPEN_ITER = 1

# 是否轻微膨胀最终 mask。
# 如果蛙人边缘经常被抠少一点，可以设为 True。
USE_FINAL_DILATE = False
FINAL_DILATE_KERNEL = 3
FINAL_DILATE_ITER = 1

# 连通域过滤。
# min_area_ratio 是相对于当前 bbox 面积的比例。
# bbox 内小物体、气泡、绳子碎片通常会被删掉。
MIN_COMPONENT_AREA_RATIO = 0.015

# 是否只保留与 bbox 中心附近相关的主要连通域。
# 对于蛙人四肢被分成多块的情况，不建议只保留最大连通域；
# 当前默认保留所有面积足够大的连通域。
KEEP_ONLY_LARGEST_COMPONENT = False

# 最终 mask 边缘羽化，用于背景替换预览 / RGBA alpha。
FEATHER_MASK = True
FEATHER_BLUR_KSIZE = 7  # 必须是奇数

# 背景替换预览颜色，BGR。
PREVIEW_BG_COLOR = (60, 120, 160)

# 显示窗口最大尺寸。
DISPLAY_MAX_WIDTH = 1800
DISPLAY_MAX_HEIGHT = 950


# =========================
# 基础工具函数
# =========================

def list_images(image_dir: str, max_images: int) -> List[Path]:
    image_dir = Path(image_dir)
    paths: List[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(image_dir.rglob(f"*{ext}"))
        paths.extend(image_dir.rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def imread_unicode(path: Path):
    """兼容 Windows 中文路径。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """兼容 Windows 中文路径。"""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix if path.suffix else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def resize_keep_aspect(img: np.ndarray,
                       max_width: int = DISPLAY_MAX_WIDTH,
                       max_height: int = DISPLAY_MAX_HEIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def safe_put_text(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def odd_kernel_size(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    if k % 2 == 0:
        k += 1
    return k


# =========================
# LabelMe bbox 读取
# =========================

def load_labelme_bboxes(json_path: Path,
                        labels_to_use: Optional[Sequence[str]] = None) -> List[Tuple[int, int, int, int]]:
    """
    读取 LabelMe json 中的 rectangle / polygon，并转成外接 bbox。

    注意：
    - 如果 labels_to_use=None，则读取所有能形成有效 bbox 的 shape。
    - 如果 json 里除了人体框还有其他标注，建议设置 BBOX_LABELS_TO_USE。
    """
    if not json_path.exists():
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    label_set = None if labels_to_use is None else set(labels_to_use)
    boxes: List[Tuple[int, int, int, int]] = []

    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label_set is not None and label not in label_set:
            continue

        pts = shape.get("points", [])
        if not pts or len(pts) < 2:
            continue

        # 只处理 rectangle / polygon。若 shape_type 缺失，也允许根据点形成 bbox。
        shape_type = shape.get("shape_type", "")
        if shape_type not in ("rectangle", "polygon", ""):
            continue

        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]

        x1 = int(np.floor(min(xs)))
        y1 = int(np.floor(min(ys)))
        x2 = int(np.ceil(max(xs)))
        y2 = int(np.ceil(max(ys)))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))

    return boxes


def get_bboxes_for_image(img_path: Path) -> List[Tuple[int, int, int, int]]:
    json_dir = Path(LABELME_JSON_DIR) if LABELME_JSON_DIR else img_path.parent
    json_path = json_dir / (img_path.stem + ".json")
    return load_labelme_bboxes(json_path, BBOX_LABELS_TO_USE)


def clip_and_expand_bbox(bbox: Tuple[int, int, int, int],
                         img_shape: Tuple[int, int, int],
                         expand_ratio: float = BBOX_EXPAND_RATIO) -> Optional[Tuple[int, int, int, int]]:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * expand_ratio))
    pad_y = int(round(bh * expand_ratio))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    return x1, y1, x2, y2


# =========================
# mask 后处理
# =========================

def fill_holes(mask: np.ndarray) -> np.ndarray:
    """填充二值 mask 内部空洞。输入/输出均为 0/255。"""
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]

    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def filter_components(mask: np.ndarray,
                      bbox: Tuple[int, int, int, int],
                      min_area_ratio: float = MIN_COMPONENT_AREA_RATIO,
                      keep_only_largest: bool = KEEP_ONLY_LARGEST_COMPONENT) -> np.ndarray:
    """
    连通域过滤：
    - 默认保留所有面积足够大的连通域，避免手臂、脚蹼、气瓶等断开后被误删。
    - 可选只保留最大连通域。
    """
    mask = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    x1, y1, x2, y2 = bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    min_area = max(8, int(round(bbox_area * float(min_area_ratio))))

    if keep_only_largest:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return np.zeros_like(mask)
        largest_label = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == largest_label] = 255
        return out

    out = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label_id] = 255

    return out


def postprocess_mask(mask: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """闭运算、开运算、填洞、连通域过滤、可选膨胀。"""
    mask = (mask > 0).astype(np.uint8) * 255

    close_k = odd_kernel_size(MORPH_CLOSE_KERNEL)
    open_k = odd_kernel_size(MORPH_OPEN_KERNEL)

    if close_k > 1 and MORPH_CLOSE_ITER > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=int(MORPH_CLOSE_ITER))

    if open_k > 1 and MORPH_OPEN_ITER > 0:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=int(MORPH_OPEN_ITER))

    mask = fill_holes(mask)
    mask = filter_components(mask, bbox)

    if USE_FINAL_DILATE and FINAL_DILATE_ITER > 0:
        dk = odd_kernel_size(FINAL_DILATE_KERNEL)
        if dk > 1:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            mask = cv2.dilate(mask, kernel_dilate, iterations=int(FINAL_DILATE_ITER))

    return (mask > 0).astype(np.uint8) * 255


# =========================
# GrabCut 抠图主逻辑
# =========================

def run_grabcut_for_bbox(img_bgr: np.ndarray,
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    对单个 bbox 执行 GrabCut，返回整图大小的前景 mask，0/255。
    """
    h, w = img_bgr.shape[:2]
    expanded = clip_and_expand_bbox(bbox, img_bgr.shape, BBOX_EXPAND_RATIO)
    if expanded is None:
        return np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = expanded

    work_img = img_bgr.copy()
    if USE_BILATERAL_FILTER:
        work_img = cv2.bilateralFilter(
            work_img,
            d=int(BILATERAL_D),
            sigmaColor=float(BILATERAL_SIGMA_COLOR),
            sigmaSpace=float(BILATERAL_SIGMA_SPACE),
        )

    grabcut_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(
            work_img,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            int(GRABCUT_ITER_COUNT),
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return np.zeros((h, w), dtype=np.uint8)

    # GrabCut 类别：
    # 0=确定背景, 1=确定前景, 2=可能背景, 3=可能前景。
    raw_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # 限制只保留 expanded bbox 内结果，防止框外误检。
    keep = np.zeros((h, w), dtype=np.uint8)
    keep[y1:y2, x1:x2] = 255
    raw_mask = cv2.bitwise_and(raw_mask, keep)

    # 后处理也基于 expanded bbox。
    final_mask = postprocess_mask(raw_mask, expanded)
    final_mask = cv2.bitwise_and(final_mask, keep)

    return final_mask


def extract_diver_mask(img_bgr: np.ndarray,
                       bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """对一张图的所有 bbox 分别 GrabCut，再合并 mask。"""
    h, w = img_bgr.shape[:2]
    final = np.zeros((h, w), dtype=np.uint8)

    for bbox in bboxes:
        m = run_grabcut_for_bbox(img_bgr, bbox)
        final = cv2.bitwise_or(final, m)

    # 多个 bbox 合并后再做一次轻度后处理，避免重叠区域碎裂。
    if len(bboxes) > 0:
        union_box = union_bboxes(bboxes, img_bgr.shape)
        if union_box is not None:
            final = postprocess_mask(final, union_box)

    return final


def union_bboxes(bboxes: List[Tuple[int, int, int, int]],
                 img_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    if not bboxes:
        return None
    h, w = img_shape[:2]
    xs1, ys1, xs2, ys2 = [], [], [], []
    for box in bboxes:
        clipped = clip_and_expand_bbox(box, img_shape, 0.0)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
    if not xs1:
        return None
    return max(0, min(xs1)), max(0, min(ys1)), min(w, max(xs2)), min(h, max(ys2))


# =========================
# 可视化与保存
# =========================

def make_alpha_mask(mask: np.ndarray) -> np.ndarray:
    """生成用于融合的 alpha，范围 0~1。"""
    alpha = (mask > 0).astype(np.uint8) * 255
    if FEATHER_MASK:
        k = odd_kernel_size(FEATHER_BLUR_KSIZE)
        if k > 1:
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha.astype(np.float32) / 255.0


def make_cutout_bgr(img_bgr: np.ndarray, mask: np.ndarray, bg_color=(0, 0, 0)) -> np.ndarray:
    alpha = make_alpha_mask(mask)[:, :, None]
    bg = np.zeros_like(img_bgr, dtype=np.float32)
    bg[:, :] = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3)
    out = img_bgr.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_rgba_cutout(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = make_alpha_mask(mask)
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return bgra


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = img_bgr.copy()
    color = np.zeros_like(vis)
    color[:, :] = (0, 0, 255)
    m = (mask > 0)[:, :, None]
    vis = np.where(m, cv2.addWeighted(vis, 0.55, color, 0.45, 0), vis)
    return vis.astype(np.uint8)


def draw_bboxes(img_bgr: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        safe_put_text(vis, f"bbox{i}", (x1, max(18, y1 - 6)), 0.5, (0, 255, 0), 1)
    return vis


def make_preview(img_bgr: np.ndarray,
                 mask: np.ndarray,
                 bboxes: List[Tuple[int, int, int, int]],
                 img_name: str) -> np.ndarray:
    """
    四栏显示：
    1. 原图+bbox
    2. mask
    3. mask 叠加
    4. 背景替换预览
    """
    src = draw_bboxes(img_bgr, bboxes)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    over = overlay_mask(draw_bboxes(img_bgr, bboxes), mask)
    replaced = make_cutout_bgr(img_bgr, mask, bg_color=PREVIEW_BG_COLOR)

    h, w = img_bgr.shape[:2]
    top_h = 38
    panel = np.zeros((h + top_h, w * 4, 3), dtype=np.uint8)
    panel[:, :] = (35, 35, 35)

    titles = [
        "Original + bbox",
        "Final diver mask",
        "Mask overlay",
        "Background replacement preview",
    ]
    imgs = [src, mask_bgr, over, replaced]

    safe_put_text(panel, f"Image: {img_name} | bboxes={len(bboxes)}", (12, 25), 0.65, (255, 255, 255), 1)

    for i, im in enumerate(imgs):
        x = i * w
        panel[top_h:top_h + h, x:x + w] = im
        safe_put_text(panel, titles[i], (x + 12, top_h + 24), 0.62, (255, 255, 255), 1)

    return panel


def save_results(out_dir: Path,
                 img_path: Path,
                 img_bgr: np.ndarray,
                 mask: np.ndarray,
                 preview: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem

    if SAVE_MASK:
        imwrite_unicode(out_dir / f"{stem}_mask.png", mask)

    if SAVE_CUTOUT_RGBA:
        rgba = make_rgba_cutout(img_bgr, mask)
        imwrite_unicode(out_dir / f"{stem}_cutout_rgba.png", rgba)

    if SAVE_PREVIEW:
        imwrite_unicode(out_dir / f"{stem}_preview.jpg", preview)


# =========================
# 主流程
# =========================

def main():
    image_paths = list_images(IMAGE_DIR, MAX_PROCESS_IMAGES)
    out_dir = Path(OUTPUT_DIR)

    if not image_paths:
        print(f"[ERROR] No images found in IMAGE_DIR: {IMAGE_DIR}")
        return

    print("bbox + GrabCut + connected components + morphology")
    print(f"IMAGE_DIR        : {IMAGE_DIR}")
    print(f"LABELME_JSON_DIR : {LABELME_JSON_DIR}")
    print(f"OUTPUT_DIR       : {OUTPUT_DIR}")
    print(f"MAX_PROCESS_IMAGES: {MAX_PROCESS_IMAGES}")
    print("Keys: any=next, s=save current, q/ESC=quit")

    for idx, img_path in enumerate(tqdm(image_paths)):
        img_bgr = imread_unicode(img_path)
        if img_bgr is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        bboxes = get_bboxes_for_image(img_path)

        if not bboxes:
            print(f"[{idx:04d}] {img_path.name}: no bbox json/shape found, skip GrabCut")
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        else:
            mask = extract_diver_mask(img_bgr, bboxes)

        preview = make_preview(img_bgr, mask, bboxes, img_path.name)

        fg_pixels = int(cv2.countNonZero(mask))
        print(f"[{idx:04d}] {img_path.name}: bboxes={len(bboxes)}, fg_pixels={fg_pixels}")

        if SAVE_RESULTS:
            save_results(out_dir, img_path, img_bgr, mask, preview)

        show = resize_keep_aspect(preview, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
        cv2.imshow("bbox_grabcut_diver_extract", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            save_results(out_dir, img_path, img_bgr, mask, preview)
            print(f"[SAVE] {img_path.name} results saved to: {out_dir}")
        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
