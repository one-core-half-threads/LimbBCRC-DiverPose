# -*- coding: utf-8 -*-
"""
debug_A_eval_mpii_pretrained_on_unlabeled_folder.py

实验 A：
    刚加载 MPII 预训练权重后，不训练，model.eval()，
    直接对半监督训练中的无标签图片做推理，并把 16 个关节点画回原图保存。

新增功能：
    从 TARGET_IMAGE_FOLDER 文件夹读取要检查的图片名，
    然后在无标签数据集 db 中按文件名或 stem 匹配对应样本，
    便于批量分析多张出现左右颠倒问题的无标签图片。

放置位置：
    建议放到你的半监督工程 pose_estimation/ 目录下运行：
        python debug_A_eval_mpii_pretrained_on_unlabeled_folder.py

这个脚本不会进入 train()，不会构建 optimizer，不会反向传播，也不会更新 BatchNorm。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


# ============================================================
# 需要你修改的参数
# ============================================================

# 工程根目录。脚本放在 pose_estimation/ 下时，默认自动取上一级目录。
# THIS_FILE = Path(__file__).resolve()
# if THIS_FILE.parent.name == "pose_estimation":
#     PROJECT_ROOT = THIS_FILE.parents[1]
# else:
#     PROJECT_ROOT = THIS_FILE.parent

CFG_FILE =  "experiments/mix_diver/hrnet/w48kp_w48limb_256x256_HeteroLimb_MPII16.yaml"

# 关节点网络的 MPII 预训练模型。
# 如果为空，则使用 yaml 里的 MODEL.PRETRAINED。
KEYPOINT_PRETRAINED = ""

# 实验 A 只检查关节点网络，limb 网络不参与输出。
# 为了避免 limb 权重路径不存在导致脚本中断，默认不加载 limb 预训练。
LOAD_LIMB_PRETRAINED = False

# 无标签数据集 split。你的 AIDataset 会读取：
# DATASET.ROOT / DiversUnlabeled / UNLABEL_IMAGE_SET / keypoint_xxx_annotation.json
UNLABEL_IMAGE_SET = "train"

# ------------------------------------------------------------
# 批量检查图片：从文件夹读取
# ------------------------------------------------------------
# 把你发现左右颠倒的无标签图片复制到这个文件夹。
# 脚本会读取该文件夹下所有图片文件名，然后到无标签数据集 db 里匹配对应样本。
TARGET_IMAGE_FOLDER = '/media/user/WKD-TF(64)/linshiceshi/input'

# 是否递归读取子文件夹。
TARGET_FOLDER_RECURSIVE = False

# 支持的图片扩展名。
TARGET_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

# True ：按文件名 stem 匹配，例如 a.jpg 可以匹配 a.png。
# False：严格按完整文件名匹配，例如 a.jpg 只能匹配 a.jpg。
MATCH_BY_STEM = True

# 如果 TARGET_IMAGE_FOLDER 不存在或里面没有图片，则是否回退到 TARGET_INDICES。
FALLBACK_TO_INDICES_IF_FOLDER_EMPTY = True

# TARGET_IMAGE_FOLDER 为空时，按无标签数据集 db 的索引检查。
TARGET_INDICES = [0, 1]

# A 实验建议使用 False：
# False：dataset is_train=False，不做随机尺度、随机旋转、180 度旋转等训练增强。
# True ：dataset is_train=True，会走训练预处理路径，但仍然 model.eval() 且不训练。
USE_TRAIN_PREPROCESS = False

# 输出目录。
OUTPUT_DIR = '/media/user/WKD-TF(64)/linshiceshi/setA'

# 是否弹窗显示。
SHOW_WINDOW = True

# 是否保存每张图的预测坐标 txt，便于后续排查。
SAVE_PRED_TXT = True

# 使用哪块 GPU。没有 GPU 会自动用 CPU。
GPU_ID = 0

# 只保留 TEST.FLIP_TEST=False，避免任何测试翻转。
FORCE_DISABLE_FLIP_TEST = True


# ============================================================
# 路径与工程导入
# ============================================================

# sys.path.insert(0, str(PROJECT_ROOT / "lib"))
# sys.path.insert(0, str(PROJECT_ROOT / "pose_estimation"))

from core.config import config, update_config  # noqa: E402
from core.inference import get_final_preds  # noqa: E402
import dataset  # noqa: E402
import models  # noqa: E402


# ============================================================
# MPII16 可视化定义
# ============================================================

JOINT_NAMES = [
    "rank", "rkne", "rhip", "lhip", "lkne", "lank",
    "root", "spine", "thorax", "head",
    "rwri", "relb", "rsho", "lsho", "lelb", "lwri",
]

SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (5, 4), (4, 3), (3, 6),
    (6, 7), (7, 8), (8, 9),
    (8, 12), (12, 11), (11, 10),
    (8, 13), (13, 14), (14, 15),
]

LIMB_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 255, 255), (0, 255, 255), (0, 255, 255),
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (255, 0, 255), (255, 0, 255), (255, 0, 255),
]

LEFT_RIGHT_PAIRS = [
    (0, 5), (1, 4), (2, 3),
    (10, 15), (11, 14), (12, 13),
]


def _set_config_for_debug():
    update_config(str(CFG_FILE))

    if KEYPOINT_PRETRAINED:
        config.MODEL.PRETRAINED = str(KEYPOINT_PRETRAINED)

    if not LOAD_LIMB_PRETRAINED:
        config.LIMB_MODEL.INIT_WEIGHTS = False
        config.LIMB_MODEL.PRETRAINED = ""

    if FORCE_DISABLE_FLIP_TEST:
        config.TEST.FLIP_TEST = False

    # 这个脚本不 resume，不加载训练 checkpoint，只加载 MODEL.PRETRAINED。
    config.TRAIN.RESUME = False

    # 保险：关闭训练数据水平翻转开关。A 实验不应该发生左右翻转。
    config.DATASET.FLIP = False

    print("=> CFG_FILE:", CFG_FILE)
    print("=> MODEL.NAME:", config.MODEL.NAME)
    print("=> MODEL.PRETRAINED:", config.MODEL.PRETRAINED)
    print("=> LIMB_MODEL.INIT_WEIGHTS:", config.LIMB_MODEL.INIT_WEIGHTS)
    print("=> DATASET.ROOT:", config.DATASET.ROOT)
    print("=> UNLABEL_IMAGE_SET:", UNLABEL_IMAGE_SET)
    print("=> TARGET_IMAGE_FOLDER:", TARGET_IMAGE_FOLDER)
    print("=> MATCH_BY_STEM:", MATCH_BY_STEM)
    print("=> USE_TRAIN_PREPROCESS:", USE_TRAIN_PREPROCESS)
    print("=> TEST.FLIP_TEST:", config.TEST.FLIP_TEST)


def _build_model(device):
    """
    注意：
    pose_hetero_limb.get_pose_net 只有 is_train=True 时才会执行 init_weights，
    所以这里必须先用 is_train=True 构建并加载预训练权重，
    然后立刻 model.eval()。
    """
    model = eval("models." + config.MODEL.NAME + ".get_pose_net")(config, is_train=True)
    model = model.to(device)
    model.eval()

    print("=> model built with is_train=True for loading pretrained weights")
    print("=> then switched to model.eval()")
    return model


def _build_unlabeled_dataset():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 这里直接用 AIDataset，因为你的 mix_mpii_ai 中无标签分支就是 AIDataset。
    # is_train=False：确定性裁剪，不做训练随机增强。
    # is_train=True ：走训练预处理路径，但仍然不训练网络。
    ds = dataset.AIDataset(
        config,
        config.DATASET.ROOT,
        UNLABEL_IMAGE_SET,
        USE_TRAIN_PREPROCESS,
        trans,
    )
    print("=> unlabeled dataset size:", len(ds.db))
    return ds


def _list_target_images_from_folder(folder):
    folder = Path(folder)

    if not folder.exists():
        print("=> WARNING: TARGET_IMAGE_FOLDER 不存在:", folder)
        return []

    files = []
    if TARGET_FOLDER_RECURSIVE:
        for ext in TARGET_IMAGE_EXTS:
            files.extend(folder.rglob("*" + ext))
    else:
        for ext in TARGET_IMAGE_EXTS:
            files.extend(folder.glob("*" + ext))

    files = sorted(set(files))
    print("=> target folder image count:", len(files))
    for p in files[:20]:
        print("   target:", p.name)
    if len(files) > 20:
        print("   ...")

    return files


def _build_db_index(ds):
    """
    建立无标签 db 的文件名索引：
        name -> [idx1, idx2, ...]
        stem -> [idx1, idx2, ...]
    """
    name_to_indices = {}
    stem_to_indices = {}

    for i, rec in enumerate(ds.db):
        image_path = Path(rec["image"])
        name = image_path.name
        stem = image_path.stem

        name_to_indices.setdefault(name, []).append(i)
        stem_to_indices.setdefault(stem, []).append(i)

    return name_to_indices, stem_to_indices


def _select_indices(ds):
    """
    优先从 TARGET_IMAGE_FOLDER 读取图片名，并在无标签 db 中匹配。
    若文件夹不存在或为空，并允许 fallback，则使用 TARGET_INDICES。
    """
    target_files = _list_target_images_from_folder(TARGET_IMAGE_FOLDER)

    if len(target_files) == 0:
        if FALLBACK_TO_INDICES_IF_FOLDER_EMPTY:
            print("=> TARGET_IMAGE_FOLDER 为空，回退使用 TARGET_INDICES:", TARGET_INDICES)
            return [i for i in TARGET_INDICES if 0 <= i < len(ds.db)]
        return []

    name_to_indices, stem_to_indices = _build_db_index(ds)

    selected = []
    missing = []
    duplicated = []

    for target_path in target_files:
        name = target_path.name
        stem = target_path.stem

        if MATCH_BY_STEM:
            matched = stem_to_indices.get(stem, [])
        else:
            matched = name_to_indices.get(name, [])

        if len(matched) == 0:
            # stem 匹配失败时，再尝试完整文件名兜底。
            if MATCH_BY_STEM:
                matched = name_to_indices.get(name, [])

        if len(matched) == 0:
            missing.append(name)
            continue

        if len(matched) > 1:
            duplicated.append((name, matched))

        # 正常情况下同名图在 db 里只出现一次；若出现多次，全部检查。
        selected.extend(matched)

    # 去重但保持顺序
    selected_unique = []
    seen = set()
    for idx in selected:
        if idx not in seen:
            selected_unique.append(idx)
            seen.add(idx)

    print("=> matched target images:", len(selected_unique))

    if missing:
        print("=> WARNING: 以下目标图片没有在无标签 db 中匹配到:")
        for name in missing[:50]:
            print("   missing:", name)
        if len(missing) > 50:
            print("   ...")

    if duplicated:
        print("=> WARNING: 以下目标图片在 db 中匹配到多个样本，将全部检查:")
        for name, inds in duplicated[:20]:
            print("   duplicated:", name, "indices=", inds)
        if len(duplicated) > 20:
            print("   ...")

    return selected_unique


def _select_heatmap(output):
    """
    兼容 pose_hetero_limb / pose_hrnet / pose_dual 输出。
    对 pose_hetero_limb 来说，输入不是 list 时 forward 会直接返回 keypoint heatmap。
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if torch.is_tensor(x) and x.ndim == 4]
        if len(tensors) == 0:
            raise RuntimeError("模型输出里没有 4D heatmap tensor")
        return tensors[-1]

    if isinstance(output, dict):
        for k in ["sup_kp", "kp", "output", "heatmap"]:
            if k in output and torch.is_tensor(output[k]):
                return output[k]

    raise TypeError("无法识别模型输出类型: {}".format(type(output)))


def _to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _decode_preds(heatmap, meta):
    center = np.asarray([_to_numpy_1d(meta["center"])[:2]], dtype=np.float32)
    scale = np.asarray([_to_numpy_1d(meta["scale"])[:2]], dtype=np.float32)

    if "rotation" in meta:
        rotation = np.asarray([float(_to_numpy_1d(meta["rotation"])[0])], dtype=np.float32)
    else:
        rotation = np.asarray([0.0], dtype=np.float32)

    preds, maxvals = get_final_preds(
        config,
        heatmap.detach().cpu().numpy(),
        center,
        scale,
        rotation,
    )
    return preds[0], maxvals[0].reshape(-1)


def _read_original_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise RuntimeError("原图读取失败: {}".format(image_path))
    return img


def _draw_bbox(img, meta):
    if "bbox_orig" not in meta:
        return

    bbox = meta["bbox_orig"]
    if torch.is_tensor(bbox):
        bbox = bbox.detach().cpu().numpy()
    bbox = np.asarray(bbox, dtype=np.float32)

    if bbox.size == 4:
        x1, y1, x2, y2 = bbox.reshape(-1).tolist()
    else:
        pts = bbox.reshape(-1, 2)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

    cv2.rectangle(
        img,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (0, 255, 0),
        2,
    )


def _valid_point(pt, img_shape):
    x, y = float(pt[0]), float(pt[1])
    h, w = img_shape[:2]
    return np.isfinite(x) and np.isfinite(y) and -10 <= x <= w + 10 and -10 <= y <= h + 10


def _draw_pose(img, preds, scores):
    # 骨架
    for i, (a, b) in enumerate(SKELETON):
        if not (_valid_point(preds[a], img.shape) and _valid_point(preds[b], img.shape)):
            continue
        xa, ya = preds[a]
        xb, yb = preds[b]
        cv2.line(
            img,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            LIMB_COLORS[i % len(LIMB_COLORS)],
            2,
            lineType=cv2.LINE_AA,
        )

    # 关节点
    for i, (x, y) in enumerate(preds):
        if not _valid_point((x, y), img.shape):
            continue

        # 右侧通道用绿色，左侧通道用红色，躯干/头用黄色，方便看左右是否交换
        if i in [0, 1, 2, 10, 11, 12]:
            color = (0, 255, 0)      # right joints
        elif i in [3, 4, 5, 13, 14, 15]:
            color = (0, 0, 255)      # left joints
        else:
            color = (0, 255, 255)    # torso/head

        xi, yi = int(round(x)), int(round(y))
        cv2.circle(img, (xi, yi), 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            "{} {}".format(i, JOINT_NAMES[i]),
            (xi + 5, yi - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def _print_lr_diagnosis(preds, scores):
    print("    left/right channel positions:")
    for r, l in LEFT_RIGHT_PAIRS:
        print(
            "      {:>2s} {:<5s}: x={:8.2f}, conf={:.4f} | {:>2s} {:<5s}: x={:8.2f}, conf={:.4f}".format(
                str(r), JOINT_NAMES[r], preds[r, 0], scores[r],
                str(l), JOINT_NAMES[l], preds[l, 0], scores[l],
            )
        )


def _save_pred_txt(out_txt, idx, image_path, preds, scores, meta):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("idx: {}\n".format(idx))
        f.write("image: {}\n".format(image_path))
        f.write("center: {}\n".format(_to_numpy_1d(meta["center"])[:2].tolist()))
        f.write("scale: {}\n".format(_to_numpy_1d(meta["scale"])[:2].tolist()))
        f.write("rotation: {}\n".format(float(_to_numpy_1d(meta.get("rotation", np.array([0])))[0])))
        f.write("\n")
        f.write("joint_index joint_name x y confidence\n")
        for i, name in enumerate(JOINT_NAMES):
            f.write("{} {} {:.4f} {:.4f} {:.6f}\n".format(
                i, name, float(preds[i, 0]), float(preds[i, 1]), float(scores[i])
            ))


def main():
    _set_config_for_debug()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(GPU_ID))
        torch.cuda.set_device(GPU_ID)
    else:
        device = torch.device("cpu")

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = _build_model(device)
    ds = _build_unlabeled_dataset()
    indices = _select_indices(ds)

    if len(indices) == 0:
        print("=> 没有找到要检查的图片，请检查 TARGET_IMAGE_FOLDER 或 TARGET_INDICES")
        return

    if SHOW_WINDOW:
        cv2.namedWindow("debug_A_eval", cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for idx in indices:
            input_tensor, target, target_weight, meta = ds[idx]
            image_path = Path(meta["image"])

            x = input_tensor.unsqueeze(0).to(device, non_blocking=True)
            output = model(x)
            heatmap = _select_heatmap(output)
            preds, scores = _decode_preds(heatmap, meta)

            img = _read_original_image(image_path)
            _draw_bbox(img, meta)
            _draw_pose(img, preds, scores)

            title = "A eval only | idx={} | {}".format(idx, image_path.name)
            cv2.putText(
                img,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

            out_path = OUTPUT_DIR / "{}_idx{}_A_eval_pred.jpg".format(image_path.stem, idx)
            cv2.imwrite(str(out_path), img)

            if SAVE_PRED_TXT:
                out_txt = OUTPUT_DIR / "{}_idx{}_A_eval_pred.txt".format(image_path.stem, idx)
                _save_pred_txt(out_txt, idx, image_path, preds, scores, meta)

            print("=> idx:", idx)
            print("   image:", image_path)
            print("   saved:", out_path)
            if SAVE_PRED_TXT:
                print("   txt:", out_txt)
            print("   meta center:", _to_numpy_1d(meta["center"])[:2])
            print("   meta scale :", _to_numpy_1d(meta["scale"])[:2])
            print("   meta rotation:", _to_numpy_1d(meta.get("rotation", np.array([0])))[0])
            _print_lr_diagnosis(preds, scores)

            if SHOW_WINDOW:
                cv2.imshow("debug_A_eval", img)
                key = cv2.waitKey(0) & 0xFF
                if key in [ord("q"), 27]:
                    break

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("=> done. output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
