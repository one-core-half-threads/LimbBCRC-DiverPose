#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Evaluate the heterogeneous semi-supervised model_best.pth.tar with the same
# validation function and metrics used during training, then draw all predicted
# poses on the original validation images.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = PROJECT_ROOT / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

# 保证无论从哪个目录运行脚本，YAML、数据集和模型中的相对路径
# 都以工程根目录为基准。
os.chdir(str(PROJECT_ROOT))

from core.config import config
from core.config import update_config
from core.config import update_dir
from core.function import validate
from core.loss import JointsMSELoss
from utils.utils import create_logger
from utils.vis import NUM_KPTS as VIS_NUM_KPTS
from utils.vis import SKELETON as VIS_SKELETON
from utils.vis import SKELETON_COLORS as VIS_SKELETON_COLORS

import dataset
import models

# Visualization-only settings for saved paper images.
VISUALIZATION_BBOX_EXPANSION = 0.50
CLIP_OUT_OF_BOUNDS = True
OUT_OF_BOUNDS_FILL_COLOR = (0, 0, 0)  # OpenCV BGR; black
PREPROCESS_BBOX_SCALE = 1.25
LOSSLESS_PNG_WRITE_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 0]

# Pose style at the 256-pixel reference width. Saved crops scale these values
# with their own width so figures displayed at one paper width look consistent.
POSE_STYLE_REFERENCE_WIDTH = 380.0
POSE_LINE_THICKNESS_AT_REFERENCE = 2.0
POSE_JOINT_RADIUS_AT_REFERENCE = 4.0
POSE_TEXT_SCALE_AT_REFERENCE = 0.45
POSE_TEXT_THICKNESS_AT_REFERENCE = 1.0
POSE_TEXT_OFFSET_AT_REFERENCE = 4.0


def _absolute_path(path):
    """
    将路径转换为绝对路径。

    相对路径默认以工程根目录 PROJECT_ROOT 为基准。
    """
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the heterogeneous semi-supervised model_best.pth.tar "
            "on the validation set using the same validate() function as "
            "pose_estimation/train.py, and save prediction visualizations."
        )
    )

    parser.add_argument(
        "--cfg",
        required=True,
        type=str,
        help="experiment YAML used for heterogeneous semi-supervised training",
    )

    parser.add_argument(
        "--model-file",
        "--model_file",
        dest="model_file",
        default="",
        type=str,
        help=(
            "path to the heterogeneous model_best.pth.tar; when omitted, "
            "use model_best.pth.tar from the experiment output directory"
        ),
    )

    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="GPU ids, for example 0 or 0,1; default: value in YAML",
    )

    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        help="number of validation data-loading workers",
    )

    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        default=None,
        type=int,
        help="validation batch size per GPU; default: TEST.BATCH_SIZE in YAML",
    )

    parser.add_argument(
        "--frequent",
        default=None,
        type=int,
        help="validation logging frequency; default: PRINT_FREQ in YAML",
    )

    parser.add_argument(
        "--data-dir",
        "--dataDir",
        dest="data_dir",
        default="",
        type=str,
        help=(
            "same optional data base directory as train.py --dataDir; "
            "omit this when DATASET.ROOT in YAML is already correct"
        ),
    )

    parser.add_argument(
        "--output-dir",
        "--modelDir",
        dest="output_dir",
        default="",
        type=str,
        help="experiment output root; default: OUTPUT_DIR in YAML",
    )

    parser.add_argument(
        "--log-dir",
        "--logDir",
        dest="log_dir",
        default="",
        type=str,
        help="log root; default: LOG_DIR in YAML",
    )

    parser.add_argument(
        "--data-format",
        "--data_format",
        dest="data_format",
        default="",
        type=str,
        help="override DATASET.DATA_FORMAT",
    )

    parser.add_argument(
        "--exp-subname",
        "--exp_subname",
        dest="exp_subname",
        default="",
        type=str,
        help="same experiment subname used during training",
    )

    parser.add_argument(
        "--vis-dir",
        "--vis_dir",
        dest="vis_dir",
        default="",
        type=str,
        help=(
            "directory used to save validation prediction images; "
            "when omitted, save under "
            "<experiment_output>/validation_pose_predictions"
        ),
    )

    # 先读取 YAML，再应用命令行覆盖参数。
    # 该处理方式与 train.py 保持一致。
    cfg_probe, _ = parser.parse_known_args()

    cfg_path = _absolute_path(cfg_probe.cfg)

    if not cfg_path.is_file():
        parser.error(
            "configuration file does not exist: {}".format(cfg_path)
        )

    update_config(str(cfg_path))

    args = parser.parse_args()
    args.cfg = str(cfg_path)

    update_dir(
        args.output_dir,
        args.log_dir,
        args.data_dir,
    )

    if args.gpus is not None:
        config.GPUS = args.gpus

    if args.workers is not None:
        config.WORKERS = args.workers

    if args.batch_size is not None:
        config.TEST.BATCH_SIZE = args.batch_size

    if args.frequent is not None:
        config.PRINT_FREQ = args.frequent

    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format

    config.EXP_SUB_NAME = args.exp_subname

    return args


def _extract_state_dict(checkpoint):
    """
    从不同形式的模型文件中提取 state_dict。

    支持：
    1. 直接保存的 state_dict；
    2. 包含 state_dict 的完整训练检查点；
    3. 包含 model_state_dict 的检查点；
    4. 直接保存的 torch.nn.Module。
    """
    if isinstance(checkpoint, torch.nn.Module):
        checkpoint = checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "unsupported checkpoint type: {}; expected a state_dict "
            "or a checkpoint containing 'state_dict'".format(
                type(checkpoint).__name__
            )
        )

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(
            "the checkpoint does not contain a non-empty state_dict"
        )

    state_dict = OrderedDict()

    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise TypeError(
                "state_dict keys must be strings"
            )

        # DataParallel 或 DDP 保存的检查点可能包含 module. 前缀。
        if key.startswith("module."):
            key = key[len("module."):]

        state_dict[key] = value

    return state_dict


def _load_model_best(model, model_file):
    """
    加载异构半监督训练生成的 model_best.pth.tar。

    使用 strict=True，确保模型结构、参数名称和参数尺寸
    与训练时使用的异构模型完全一致。
    """
    try:
        checkpoint = torch.load(
            str(model_file),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # 兼容没有 weights_only 参数的旧版本 PyTorch。
        checkpoint = torch.load(
            str(model_file),
            map_location="cpu",
        )

    state_dict = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "model_best is incompatible with MODEL.NAME='{}' and the "
            "supplied YAML. Use the exact training YAML and the complete "
            "heterogeneous-model checkpoint.\n{}".format(
                config.MODEL.NAME,
                exc,
            )
        )

    return len(state_dict)


def _parse_gpu_ids(gpu_text):
    """
    将 GPU 字符串转换为整数列表。

    例如：
        "0"       -> [0]
        "0,1"     -> [0, 1]
        "4,5,6,7" -> [4, 5, 6, 7]
    """
    try:
        gpu_ids = [
            int(item.strip())
            for item in str(gpu_text).split(",")
        ]
    except ValueError:
        raise ValueError(
            "invalid GPU list {!r}; use e.g. 0 or 0,1".format(
                gpu_text
            )
        )

    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(
            "GPU ids must be non-negative integers"
        )

    return gpu_ids


def _read_original_image(image_file):
    """
    读取一张验证集原始图片。

    返回 OpenCV 使用的 BGR 图像。
    """
    image_file = str(image_file)

    if config.DATASET.DATA_FORMAT == "zip" or "@/" in image_file:
        from utils import zipreader

        return zipreader.imread(
            image_file,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )

    return cv2.imread(
        image_file,
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )


def _relative_visualization_path(image_file):
    """
    构造可视化结果的相对保存路径，并尽量保留原图目录结构。
    """
    image_file = str(image_file)

    # 压缩包图像路径示例：
    # /data/.../images.zip@/images/xxx.jpg
    #
    # 转换为：
    # images/xxx.jpg
    if "@/" in image_file:
        inner_path = image_file.split("@/", 1)[1]
        relative_path = Path(inner_path)

    else:
        image_path = Path(image_file).expanduser().resolve()

        dataset_root = Path(config.DATASET.ROOT).expanduser()

        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root

        dataset_root = dataset_root.resolve()

        try:
            relative_path = image_path.relative_to(dataset_root)
        except ValueError:
            relative_path = Path(image_path.name)

    # cv2.imwrite 需要可识别的图片扩展名。
    valid_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    if relative_path.suffix.lower() not in valid_suffixes:
        relative_path = relative_path.with_suffix(".png")

    return relative_path.with_suffix(".png")


def _expanded_person_crop_bounds(image_shape, person_crops):
    image_height, image_width = image_shape[:2]
    expanded_bounds = []

    for center, model_scale in person_crops:
        center = np.asarray(center, dtype=np.float64).reshape(-1)
        model_scale = np.asarray(model_scale, dtype=np.float64).reshape(-1)
        if center.size < 2 or model_scale.size < 1:
            continue

        square_side = (
            float(np.max(model_scale[:2]))
            * 200.0
            * (1.0 + VISUALIZATION_BBOX_EXPANSION)
            / PREPROCESS_BBOX_SCALE
        )
        if (
            not np.isfinite(square_side)
            or square_side <= 0
            or not np.all(np.isfinite(center[:2]))
        ):
            continue

        pixel_side = int(np.ceil(square_side))
        left = int(np.floor(center[0] - pixel_side / 2.0))
        top = int(np.floor(center[1] - pixel_side / 2.0))
        right = left + pixel_side
        bottom = top + pixel_side

        if CLIP_OUT_OF_BOUNDS:
            left = max(0, left)
            top = max(0, top)
            right = min(image_width, right)
            bottom = min(image_height, bottom)

        if right > left and bottom > top:
            expanded_bounds.append((left, top, right, bottom))

    if not expanded_bounds:
        return None

    return (
        min(bounds[0] for bounds in expanded_bounds),
        min(bounds[1] for bounds in expanded_bounds),
        max(bounds[2] for bounds in expanded_bounds),
        max(bounds[3] for bounds in expanded_bounds),
    )


def _crop_canvas_with_boundary_mode(canvas, crop_bounds):
    left, top, right, bottom = crop_bounds
    image_height, image_width = canvas.shape[:2]

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image_width, right)
    source_bottom = min(image_height, bottom)
    if source_right <= source_left or source_bottom <= source_top:
        return None

    crop = canvas[source_top:source_bottom, source_left:source_right]
    if CLIP_OUT_OF_BOUNDS:
        return crop

    return cv2.copyMakeBorder(
        crop,
        max(0, -top),
        max(0, bottom - image_height),
        max(0, -left),
        max(0, right - image_width),
        cv2.BORDER_CONSTANT,
        value=OUT_OF_BOUNDS_FILL_COLOR,
    )


def _adaptive_pose_style(image_width):
    scale = max(float(image_width), 1.0) / POSE_STYLE_REFERENCE_WIDTH

    def scaled_pixels(reference_value):
        return max(1, int(np.floor(reference_value * scale + 0.5)))

    return {
        "line_thickness": scaled_pixels(
            POSE_LINE_THICKNESS_AT_REFERENCE
        ),
        "joint_radius": scaled_pixels(POSE_JOINT_RADIUS_AT_REFERENCE),
        "text_scale": max(0.1, POSE_TEXT_SCALE_AT_REFERENCE * scale),
        "text_thickness": scaled_pixels(
            POSE_TEXT_THICKNESS_AT_REFERENCE
        ),
        "text_offset": scaled_pixels(POSE_TEXT_OFFSET_AT_REFERENCE),
    }


def _draw_pose_on_image_adaptive(
    image,
    keypoints,
    joints_vis,
    pose_style,
    draw_invisible=False,
):
    if keypoints is None:
        return image

    def visible(joint_id):
        if joint_id >= len(joints_vis):
            return bool(draw_invisible)
        return (
            bool(float(joints_vis[joint_id][0]) > 0)
            or bool(draw_invisible)
        )

    def valid_point(joint):
        x, y = float(joint[0]), float(joint[1])
        if (
            x < 0
            or y < 0
            or not np.isfinite(x)
            or not np.isfinite(y)
        ):
            return None
        return int(round(x)), int(round(y))

    num_joints = min(len(keypoints), int(VIS_NUM_KPTS))
    for limb_index, (joint_a, joint_b) in enumerate(VIS_SKELETON):
        if joint_a >= num_joints or joint_b >= num_joints:
            continue
        if not visible(joint_a) or not visible(joint_b):
            continue
        point_a = valid_point(keypoints[joint_a])
        point_b = valid_point(keypoints[joint_b])
        if point_a is None or point_b is None:
            continue
        color = VIS_SKELETON_COLORS[
            limb_index % len(VIS_SKELETON_COLORS)
        ]
        cv2.line(
            image,
            point_a,
            point_b,
            color,
            pose_style["line_thickness"],
            cv2.LINE_AA,
        )

    for joint_id in range(num_joints):
        if not visible(joint_id):
            continue
        point = valid_point(keypoints[joint_id])
        if point is None:
            continue
        x, y = point
        cv2.circle(
            image,
            (x, y),
            pose_style["joint_radius"],
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )
        offset = pose_style["text_offset"]
        cv2.putText(
            image,
            str(joint_id),
            (x + offset, y - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            pose_style["text_scale"],
            (255, 255, 255),
            pose_style["text_thickness"],
            cv2.LINE_AA,
        )
    return image


def _save_validation_prediction_images(
    val_dataset,
    all_preds_file,
    visualization_dir,
    logger,
):
    """
    将验证集预测姿态绘制到原图上。

    MPII 风格的数据集中，一条数据库记录对应一个人员实例。
    因此，同一张图片中存在多个人时，会有多条记录共享相同的图片路径。

    本函数先按照原图路径对预测结果进行分组，然后把同一张图中的
    所有人员预测姿态绘制到同一张图片上。

    骨架连接方式、骨架颜色、关节点颜色、关节点半径、线宽和编号
    都调用 utils.vis._draw_pose_on_image()，与 YAML 的 DEBUG
    可视化代码保持一致。
    """
    all_preds_file = Path(all_preds_file)

    if not all_preds_file.is_file():
        raise FileNotFoundError(
            "validate() did not produce the expected prediction file: "
            "{}".format(all_preds_file)
        )

    all_preds = np.load(str(all_preds_file))

    if len(all_preds) != len(val_dataset.db):
        raise RuntimeError(
            "prediction count ({}) does not match validation record "
            "count ({})".format(
                len(all_preds),
                len(val_dataset.db),
            )
        )

    # 验证集数据库中一条记录对应一个人物实例。
    # 将相同图片路径的所有人物预测结果放在同一个列表中。
    grouped_predictions = OrderedDict()
    grouped_person_crops = OrderedDict()

    for sample_index, db_record in enumerate(val_dataset.db):
        image_file = str(db_record["image"])

        grouped_predictions.setdefault(
            image_file,
            [],
        ).append(
            all_preds[sample_index, :, 0:2].copy()
        )
        grouped_person_crops.setdefault(
            image_file,
            [],
        ).append(
            (
                np.asarray(db_record["center"], dtype=np.float64).copy(),
                np.asarray(db_record["scale"], dtype=np.float64).copy(),
            )
        )

    visualization_dir = Path(visualization_dir)
    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 预测结果没有标签可见性标志。
    # 将所有关节点可见性设为 1，使绘制函数显示全部预测关节点。
    draw_vis = np.ones(
        (
            int(config.MODEL.NUM_JOINTS),
            1,
        ),
        dtype=np.float32,
    )

    saved_count = 0
    failed_images = []
    total_images = len(grouped_predictions)

    for image_index, (
        image_file,
        people_predictions,
    ) in enumerate(grouped_predictions.items()):

        image = _read_original_image(image_file)

        if image is None:
            failed_images.append(image_file)

            logger.warning(
                "=> failed to read visualization image: %s",
                image_file,
            )
            continue

        crop_bounds = _expanded_person_crop_bounds(
            image.shape,
            grouped_person_crops[image_file],
        )
        if crop_bounds is None:
            failed_images.append(image_file)
            logger.warning(
                "=> invalid person box; skipped visualization crop: %s",
                image_file,
            )
            continue

        pose_style = _adaptive_pose_style(
            crop_bounds[2] - crop_bounds[0]
        )
        canvas = image.copy()

        # 将同一张图片中的所有人物姿态绘制在同一张图上。
        for person_prediction in people_predictions:
            _draw_pose_on_image_adaptive(
                canvas,
                person_prediction,
                draw_vis,
                pose_style,
                draw_invisible=True,
            )

        cropped_canvas = _crop_canvas_with_boundary_mode(
            canvas,
            crop_bounds,
        )
        if cropped_canvas is None:
            failed_images.append(image_file)
            logger.warning(
                "=> person crop does not overlap the source image: %s",
                image_file,
            )
            continue

        save_path = (
            visualization_dir
            / _relative_visualization_path(image_file)
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not cv2.imwrite(
            str(save_path),
            cropped_canvas,
            LOSSLESS_PNG_WRITE_PARAMS,
        ):
            failed_images.append(image_file)

            logger.warning(
                "=> failed to write visualization image: %s",
                save_path,
            )
            continue

        saved_count += 1

        if (
            (image_index + 1) % 100 == 0
            or image_index + 1 == total_images
        ):
            logger.info(
                "=> visualization progress: %d/%d original images",
                image_index + 1,
                total_images,
            )

    logger.info(
        "=> saved %d validation prediction image(s) to %s",
        saved_count,
        visualization_dir,
    )

    if failed_images:
        logger.warning(
            "=> %d image(s) could not be visualized",
            len(failed_images),
        )


def main():
    args = parse_args()

    # 此脚本只允许评估完整的异构半监督模型。
    # 不允许错误地使用监督学习的单分支 pose_hrnet 模型。
    if config.MODEL.NAME != "pose_hetero_limb":
        raise ValueError(
            "this visualization evaluator expects "
            "MODEL.NAME=pose_hetero_limb, but the supplied YAML contains "
            "MODEL.NAME={!r}. Use the exact heterogeneous semi-supervised "
            "training YAML and its model_best.pth.tar.".format(
                config.MODEL.NAME
            )
        )

    logger, final_output_dir, tb_log_dir = create_logger(
        config,
        args.cfg,
        "valid",
    )

    if args.model_file:
        model_file = _absolute_path(args.model_file)
    else:
        model_file = (
            Path(final_output_dir)
            / "model_best.pth.tar"
        )

    config.TEST.MODEL_FILE = str(model_file)

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    if not model_file.is_file():
        raise FileNotFoundError(
            "model_best checkpoint does not exist: {}\n"
            "Pass its path with --model-file.".format(
                model_file
            )
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required because the training validation function "
            "moves targets and loss computation to GPU."
        )

    gpu_ids = _parse_gpu_ids(config.GPUS)

    gpu_count = torch.cuda.device_count()

    unavailable = [
        gpu_id
        for gpu_id in gpu_ids
        if gpu_id >= gpu_count
    ]

    if unavailable:
        raise RuntimeError(
            "requested GPU ids {} but only {} CUDA device(s) "
            "are visible".format(
                unavailable,
                gpu_count,
            )
        )

    torch.cuda.set_device(gpu_ids[0])

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # 根据 YAML 中的 MODEL.NAME 构建完整异构半监督模型：
    #
    # models.pose_hetero_limb.get_pose_net(...)
    model = eval(
        "models."
        + config.MODEL.NAME
        + ".get_pose_net"
    )(
        config,
        is_train=False,
    )

    # 加载异构半监督训练生成的完整 model_best.pth.tar。
    num_tensors = _load_model_best(
        model,
        model_file,
    )

    logger.info(
        "=> loaded {} tensors from {}".format(
            num_tensors,
            model_file,
        )
    )

    model = torch.nn.DataParallel(
        model,
        device_ids=gpu_ids,
    ).cuda()

    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT,
        cfg=config,
    ).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # 验证集构建方式与原 evaluate_model_best.py 保持一致。
    valid_dataset = eval(
        "dataset."
        + config.DATASET.TEST_DATASET
    )(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=(
            config.TEST.BATCH_SIZE
            * len(gpu_ids)
        ),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )

    logger.info(
        "=> evaluating {} validation samples with {}".format(
            len(valid_dataset),
            model_file.name,
        )
    )

    logger.info(
        "=> validation dataset: dataset={}, root={}, split={}".format(
            config.DATASET.TEST_DATASET,
            config.DATASET.ROOT,
            config.DATASET.TEST_SET,
        )
    )

    # 原封不动地调用 train.py 使用的 validate()。
    #
    # 因此验证指标、PCKh 计算、日志格式、pred.mat 和 all_preds.npy
    # 均与原 evaluate_model_best.py 一致。
    #
    # 对 pose_hetero_limb 而言，验证阶段模型输出的是关节点分支
    # keypoint_net 的预测热度图。
    perf_indicator = validate(
        config,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        final_output_dir,
        tb_log_dir,
    )

    # 指定预测图片保存目录。
    if args.vis_dir:
        visualization_dir = _absolute_path(
            args.vis_dir
        )
    else:
        visualization_dir = (
            Path(final_output_dir)
            / "validation_pose_predictions"
        )

    # validate() 会在 final_output_dir 下生成 all_preds.npy。
    # 读取该文件，把预测结果绘制回验证集原始图片。
    _save_validation_prediction_images(
        valid_dataset,
        Path(final_output_dir) / "all_preds.npy",
        visualization_dir,
        logger,
    )

    return perf_indicator


if __name__ == "__main__":
    main()

# python pose_estimation/evaluate_model_best_vis.py \
#   --cfg experiments/mix_diver/hrnet/w48kp_w48limb_256x256_HeteroLimb_MPII16.yaml \
#   --model-file output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-flip-90-uir180*2-7-B2B-0.1/model_best.pth.tar \
#   --gpus 0 \
#   --vis-dir output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-flip-90-uir180*2-7-B2B-0.1/vis_results
