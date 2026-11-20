# -*- coding: utf-8 -*-
"""
visualize_mpii_labels_direct.py

读取 MPII/HRNet 格式标注 json，将 center/scale 对应的包围框和 16 个关节点画在图像上显示。

使用方式：
1. 把本文件放到 HRNet 工程根目录下，或直接修改下面“需要你修改的参数”。
2. 运行：
       python visualize_mpii_labels_direct.py

按键说明：
    n / Space / Enter / 其他键 : 下一张
    p                         : 上一张
    s                         : 保存当前可视化图
    q / Esc                   : 退出

说明：
- MPII/HRNet 标注 json 通常没有显式 bbox，而是用 center + scale 表示人体框。
- BOX_MODE = "raw"   ：显示 json 原始 center/scale 对应的框。
- BOX_MODE = "hrnet" ：显示 HRNet mpii.py 训练时实际使用的裁剪框。
- BOX_MODE = "both"  ：同时显示两种框。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ============================================================
# 需要你修改的参数
# ============================================================

# MPII 数据集根目录。默认图片会从 DATA_ROOT / "images" 下读取。
DATA_ROOT = Path("/data/WKD/MultiAugs/data/mpii")

# HRNet 格式 MPII 标注文件。
ANN_FILE = DATA_ROOT / "annot" / "train.json"
# ANN_FILE = DATA_ROOT / "annot" / "valid.json"

# 如果你的图片不在 DATA_ROOT/images 下，可以在这里直接指定图片目录。
# 例如：IMAGE_DIR = Path("/your/path/to/mpii/images")
IMAGE_DIR: Optional[Path] = None

# 可视化结果保存目录。
OUTPUT_DIR = Path("t")

# bbox 显示方式："raw" / "hrnet" / "both"
# raw   : json 原始 center/scale 框
# hrnet : HRNet mpii.py 训练实际裁剪框，即 center_y += 15 * scale, scale *= 1.25, center -= 1
# both  : 两种都显示
BOX_MODE = "both"

# 是否画出 joints_vis <= 0 但坐标有效的不可见关节点。
DRAW_INVISIBLE = True

# MPII 原始标注通常是 1-based 坐标，HRNet mpii.py 读取时会减 1。
# 如果你的 json 已经是 0-based 坐标，把这里改成 False。
SUBTRACT_ONE = True

# 从第几条标注开始显示。
START_INDEX = 0

# 最多显示多少条。-1 表示全部。
MAX_NUM = -1

# 是否自动保存所有显示过的图片。
SAVE_ALL = False

# 是否不弹出窗口，只保存图片。服务器无图形界面时可设为 True。
NO_SHOW = False

# imshow 窗口最大尺寸。
MAX_WINDOW_WIDTH = 1600
MAX_WINDOW_HEIGHT = 1000


# ============================================================
# MPII 16 关节点定义和显示颜色
# ============================================================

# MPII 16 关节点顺序：
# 0 rank, 1 rkne, 2 rhip, 3 lhip, 4 lkne, 5 lank,
# 6 pelvis/root, 7 thorax/spine, 8 upper neck, 9 head top,
# 10 rwri, 11 relb, 12 rsho, 13 lsho, 14 lelb, 15 lwri
JOINT_NAMES = [
    "rank", "rkne", "rhip", "lhip", "lkne", "lank",
    "root", "spine", "thorax", "head",
    "rwri", "relb", "rsho", "lsho", "lelb", "lwri",
]

SKELETON = [
    (0, 1), (1, 2), (2, 6),       # 右腿
    (5, 4), (4, 3), (3, 6),       # 左腿
    (6, 7), (7, 8), (8, 9),       # 躯干 + 头
    (8, 12), (12, 11), (11, 10),  # 右臂
    (8, 13), (13, 14), (14, 15),  # 左臂
]

# OpenCV 使用 BGR 颜色。
LIMB_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (255, 0, 255), (255, 0, 255), (255, 0, 255),
]

COLOR_RAW_BOX = (0, 255, 0)          # 绿色：原始 center/scale 框
COLOR_HRNET_BOX = (0, 255, 255)      # 黄色：HRNet 训练实际裁剪框
COLOR_VISIBLE_JOINT = (0, 0, 255)    # 红色：可见关节点
COLOR_INVISIBLE_JOINT = (180, 180, 180)
COLOR_TEXT = (255, 255, 255)
PIXEL_STD = 200.0


def load_json_records(ann_path: Path) -> List[Dict[str, Any]]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # 兼容少数自定义 json 格式。
    for key in ["annotations", "annolist", "root", "data"]:
        if key in data and isinstance(data[key], list):
            return data[key]

    raise ValueError(f"无法识别标注 json 格式: {ann_path}")


def find_image_path(record: Dict[str, Any]) -> Optional[Path]:
    image_name = record.get("image", record.get("img_paths", record.get("file_name", "")))
    if not image_name:
        return None

    image_name = str(image_name)
    image_path = Path(image_name)

    candidates = []

    if image_path.is_absolute():
        candidates.append(image_path)

    if IMAGE_DIR is not None:
        candidates.append(IMAGE_DIR / image_name)
        candidates.append(IMAGE_DIR / image_path.name)

    candidates.extend([
        DATA_ROOT / "images" / image_name,
        DATA_ROOT / "images" / image_path.name,
        DATA_ROOT / image_name,
        DATA_ROOT / image_path.name,
    ])

    for p in candidates:
        if p.exists():
            return p

    return None


def as_scale_array(scale_value: Any) -> np.ndarray:
    """
    HRNet MPII json 中 scale 通常是一个 float；
    如果是自定义格式 [sx, sy]，这里也兼容。
    """
    s = np.asarray(scale_value, dtype=np.float32)
    if s.ndim == 0:
        return np.array([float(s), float(s)], dtype=np.float32)
    if s.size == 1:
        v = float(s.reshape(-1)[0])
        return np.array([v, v], dtype=np.float32)
    return s.reshape(-1)[:2].astype(np.float32)


def center_scale_to_box(center: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32).reshape(2)
    s = np.asarray(scale, dtype=np.float32).reshape(2)

    w = float(s[0]) * PIXEL_STD
    h = float(s[1]) * PIXEL_STD

    x1 = c[0] - w * 0.5
    y1 = c[1] - h * 0.5
    x2 = c[0] + w * 0.5
    y2 = c[1] + h * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_boxes_from_record(record: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    返回：
        raw_box   : json 原始 center/scale 对应框。
        hrnet_box : HRNet mpii.py 训练实际使用的 center/scale 对应框。

    HRNet lib/dataset/mpii.py 中的核心逻辑通常是：
        c = np.array(a['center'], dtype=float)
        s = np.array([a['scale'], a['scale']], dtype=float)
        if c[0] != -1:
            c[1] = c[1] + 15 * s[1]
            s = s * 1.25
        c = c - 1
        joints[:, 0:2] = joints[:, 0:2] - 1
    """
    if "center" not in record or "scale" not in record:
        return None, None

    c_raw = np.asarray(record["center"], dtype=np.float32).reshape(2)
    s_raw = as_scale_array(record["scale"])

    # raw 框：只做 1-based -> 0-based 修正，便于和原图像坐标对齐。
    c_raw_for_draw = c_raw.copy()
    if SUBTRACT_ONE:
        c_raw_for_draw -= 1.0
    raw_box = center_scale_to_box(c_raw_for_draw, s_raw)

    # HRNet 训练实际裁剪框：先执行 y 补偿和 scale 扩大，再做 1-based -> 0-based。
    c_hrnet = c_raw.copy()
    s_hrnet = s_raw.copy()
    if c_hrnet[0] != -1:
        c_hrnet[1] = c_hrnet[1] + 15.0 * s_hrnet[1]
        s_hrnet = s_hrnet * 1.25
    if SUBTRACT_ONE:
        c_hrnet -= 1.0
    hrnet_box = center_scale_to_box(c_hrnet, s_hrnet)

    return raw_box, hrnet_box


def get_joints_from_record(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(record.get("joints", []), dtype=np.float32)
    if joints.size == 0:
        joints = np.zeros((0, 2), dtype=np.float32)

    if joints.ndim == 1:
        joints = joints.reshape(-1, 2)
    if joints.shape[1] > 2:
        joints = joints[:, :2]

    if SUBTRACT_ONE and joints.size > 0:
        joints = joints.copy()
        joints[:, :2] -= 1.0

    joints_vis = np.asarray(record.get("joints_vis", np.ones((len(joints), 1))), dtype=np.float32)
    if joints_vis.ndim == 1:
        joints_vis = joints_vis.reshape(-1, 1)

    if joints_vis.shape[0] != joints.shape[0]:
        fixed = np.zeros((joints.shape[0], 1), dtype=np.float32)
        n = min(joints_vis.shape[0], joints.shape[0])
        fixed[:n, 0] = joints_vis.reshape(joints_vis.shape[0], -1)[:n, 0]
        joints_vis = fixed

    vis = joints_vis.reshape(joints_vis.shape[0], -1)[:, 0] > 0
    return joints, vis


def is_valid_point(pt: Sequence[float], image_shape: Tuple[int, int, int]) -> bool:
    x, y = float(pt[0]), float(pt[1])
    h, w = image_shape[:2]

    if not np.isfinite(x) or not np.isfinite(y):
        return False

    # 明显无效点不画；允许轻微越界，便于检查边界附近标注。
    if x < -5 or y < -5 or x > w + 5 or y > h + 5:
        return False

    return True


def draw_bbox(
    img: np.ndarray,
    box_xyxy: np.ndarray,
    color: Tuple[int, int, int],
    text: str,
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy.tolist()]

    x1c, y1c = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2c, y2c = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

    cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, thickness)
    cv2.putText(
        img,
        text,
        (x1c, max(15, y1c - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def draw_pose(img: np.ndarray, joints: np.ndarray, joints_vis: np.ndarray) -> None:
    num_joints = min(len(joints), len(JOINT_NAMES))

    # 先画骨架线。
    for limb_idx, (a, b) in enumerate(SKELETON):
        if a >= num_joints or b >= num_joints:
            continue

        a_draw = bool(joints_vis[a]) or DRAW_INVISIBLE
        b_draw = bool(joints_vis[b]) or DRAW_INVISIBLE
        if not (a_draw and b_draw):
            continue
        if not (is_valid_point(joints[a], img.shape) and is_valid_point(joints[b], img.shape)):
            continue

        xa, ya = joints[a]
        xb, yb = joints[b]
        color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
        cv2.line(
            img,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    # 再画关节点和编号。
    for i in range(num_joints):
        if not is_valid_point(joints[i], img.shape):
            continue

        visible = bool(joints_vis[i])
        if not visible and not DRAW_INVISIBLE:
            continue

        x, y = joints[i]
        x_i, y_i = int(round(x)), int(round(y))
        color = COLOR_VISIBLE_JOINT if visible else COLOR_INVISIBLE_JOINT

        if visible:
            cv2.circle(img, (x_i, y_i), 4, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(img, (x_i, y_i), 4, color, 1, lineType=cv2.LINE_AA)

        cv2.putText(
            img,
            str(i),
            (x_i + 5, y_i - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )


def draw_legend(img: np.ndarray) -> None:
    lines = []
    if BOX_MODE in ["raw", "both"]:
        lines.append(("raw center/scale box", COLOR_RAW_BOX))
    if BOX_MODE in ["hrnet", "both"]:
        lines.append(("HRNet train crop box", COLOR_HRNET_BOX))
    lines.append(("visible joint", COLOR_VISIBLE_JOINT))
    if DRAW_INVISIBLE:
        lines.append(("invisible/unweighted joint", COLOR_INVISIBLE_JOINT))

    x0, y0 = 10, 22
    for i, (text, color) in enumerate(lines):
        y = y0 + i * 22
        cv2.rectangle(img, (x0, y - 12), (x0 + 16, y + 4), color, -1)
        cv2.putText(
            img,
            text,
            (x0 + 24, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT,
            1,
            lineType=cv2.LINE_AA,
        )


def fit_for_display(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(float(MAX_WINDOW_WIDTH) / max(w, 1),
                float(MAX_WINDOW_HEIGHT) / max(h, 1),
                1.0)
    if scale >= 1.0:
        return img

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def safe_output_name(index: int, image_path: Path) -> str:
    return f"{index:06d}_{image_path.stem}_label_vis{image_path.suffix}"


def save_image(index: int, image_path: Path, vis_img: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / safe_output_name(index, image_path)
    cv2.imwrite(str(out_path), vis_img)
    return out_path


def visualize_one(record: Dict[str, Any], image_path: Path, index: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise RuntimeError(f"图像读取失败: {image_path}")

    vis_img = img.copy()

    raw_box, hrnet_box = get_boxes_from_record(record)
    joints, joints_vis = get_joints_from_record(record)

    if BOX_MODE in ["raw", "both"] and raw_box is not None:
        draw_bbox(vis_img, raw_box, COLOR_RAW_BOX, "raw center/scale")
    if BOX_MODE in ["hrnet", "both"] and hrnet_box is not None:
        draw_bbox(vis_img, hrnet_box, COLOR_HRNET_BOX, "HRNet train crop")

    draw_pose(vis_img, joints, joints_vis)

    title = f"idx={index}  image={image_path.name}"
    cv2.putText(
        vis_img,
        title,
        (10, max(25, vis_img.shape[0] - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        COLOR_TEXT,
        2,
        lineType=cv2.LINE_AA,
    )

    draw_legend(vis_img)
    return vis_img


def main() -> None:
    if BOX_MODE not in ["raw", "hrnet", "both"]:
        raise ValueError('BOX_MODE 只能是 "raw"、"hrnet" 或 "both"。')

    records = load_json_records(ANN_FILE)
    total = len(records)

    start = max(0, int(START_INDEX))
    end = total if MAX_NUM < 0 else min(total, start + int(MAX_NUM))

    print(f"=> 标注文件: {ANN_FILE}")
    print(f"=> 标注数量: {total}")
    print(f"=> 显示范围: [{start}, {end})")
    print(f"=> BOX_MODE: {BOX_MODE}")
    print(f"=> DRAW_INVISIBLE: {DRAW_INVISIBLE}")
    print(f"=> 坐标处理: {'MPII 1-based -> 0-based' if SUBTRACT_ONE else '保持 json 原坐标，不减 1'}")
    print(f"=> 输出目录: {OUTPUT_DIR}")

    if not NO_SHOW:
        cv2.namedWindow("MPII label visualization", cv2.WINDOW_NORMAL)

    idx = start
    while start <= idx < end:
        record = records[idx]
        image_path = find_image_path(record)

        if image_path is None:
            print(f"[跳过] 找不到图片 idx={idx}, image={record.get('image')}")
            idx += 1
            continue

        try:
            vis_img = visualize_one(record, image_path, idx)
        except Exception as exc:
            print(f"[跳过] 可视化失败 idx={idx}, image={image_path}, error={exc}")
            idx += 1
            continue

        if SAVE_ALL or NO_SHOW:
            out_path = save_image(idx, image_path, vis_img)
            print(f"[{idx + 1}/{total}] saved: {out_path}")
            idx += 1
            continue

        show_img = fit_for_display(vis_img)
        cv2.imshow("MPII label visualization", show_img)

        print(
            f"[{idx + 1}/{total}] {image_path.name} | "
            "按 n/Space/Enter 下一张，p 上一张，s 保存，q/Esc 退出"
        )

        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), 27]:
            break

        if key == ord("s"):
            out_path = save_image(idx, image_path, vis_img)
            print(f"    saved: {out_path}")
            # 保存后仍停留在当前图，方便继续检查。
            continue

        if key == ord("p"):
            idx = max(start, idx - 1)
            continue

        # n / Space / Enter / 其他键都进入下一张。
        idx += 1

    if not NO_SHOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
