# -*- coding: utf-8 -*-
"""
从 pose_hetero_limb 检查点中分别导出关节点网络和肢体网络参数。

保存格式与 tool/convert_pose_dual_to_hrnet.py 一致：
    torch.save(new_state_dict, output_path)

也就是保存裸的 OrderedDict，不再套 {'state_dict': ...}。

使用方式：
    直接修改本文件末尾的 checkpoint_path、output_kp_path、output_limb_path，
    然后运行：
        python split_hetero_limb_checkpoint_to_hrnet_noarg.py

默认输出文件名建议为：
    checkpoint_kp.pth
    checkpoint_limb.pth
"""

from collections import OrderedDict
from pathlib import Path

import torch


def _strip_prefix_once(key: str, prefix: str) -> str:
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def clean_state_key(key: str) -> str:
    """
    兼容 DataParallel / DDP / 某些外层 model 包装保存出来的前缀。

    常见形式：
        module.keypoint_net.conv1.weight
        model.keypoint_net.conv1.weight
        module.model.keypoint_net.conv1.weight
        keypoint_net.conv1.weight
    """
    changed = True
    while changed:
        old = key
        for prefix in ("module.", "model.", "net."):
            key = _strip_prefix_once(key, prefix)
        changed = (key != old)
    return key


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and hasattr(checkpoint["model"], "items"):
        state_dict = checkpoint["model"]
    else:
        # final_state.pth.tar / model_best.pth.tar 有时可能直接就是裸 state_dict。
        state_dict = checkpoint

    if not hasattr(state_dict, "items"):
        raise RuntimeError(
            f"{path} 不是可识别的 PyTorch state_dict/checkpoint。"
        )

    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[clean_state_key(str(k))] = v

    return cleaned


def extract_branch_state_dict(state_dict, net):
    """
    提取指定分支参数，并去掉分支前缀，使其可直接作为单独 HRNet 权重加载。

    例如：
        keypoint_net.conv1.weight -> conv1.weight
        limb_net.final_layer.bias -> final_layer.bias
    """
    assert net in ["keypoint_net", "limb_net"], "net must be keypoint_net or limb_net"

    prefix = net + "."
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v

    if len(new_state_dict) == 0:
        sample_keys = list(state_dict.keys())[:30]
        raise RuntimeError(
            f"No parameters with prefix '{prefix}' found. "
            f"Please check whether this is a pose_hetero_limb checkpoint.\n"
            f"First 30 keys:\n" + "\n".join(sample_keys)
        )

    return new_state_dict


def save_branch(new_state_dict, output_path, net):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 与 convert_pose_dual_to_hrnet.py 保持一致：只保存裸 OrderedDict。
    torch.save(new_state_dict, str(output_path))

    print(f"Saved converted HRNet weights to: {output_path}")
    print(f"Extracted network: {net}")
    print(f"Number of tensors: {len(new_state_dict)}")

    for name in ["conv1.weight", "bn1.weight", "final_layer.weight", "final_layer.bias"]:
        if name in new_state_dict:
            print(name, tuple(new_state_dict[name].shape))
        else:
            print("Missing:", name)

    print("-" * 60)


def convert(checkpoint_path, output_kp_path, output_limb_path):
    checkpoint_path = Path(checkpoint_path)
    output_kp_path = Path(output_kp_path)
    output_limb_path = Path(output_limb_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_state_dict(str(checkpoint_path))
    print(f"Total tensors in checkpoint state_dict: {len(state_dict)}")

    kp_state_dict = extract_branch_state_dict(state_dict, "keypoint_net")
    limb_state_dict = extract_branch_state_dict(state_dict, "limb_net")

    save_branch(kp_state_dict, output_kp_path, "keypoint_net")
    save_branch(limb_state_dict, output_limb_path, "limb_net")

    print("Done.")


if __name__ == "__main__":
    # pose_hetero_limb checkpoint path
    checkpoint_path = "output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-dfbra_aug/model_best.pth.tar"

    # Output HRNet weight paths
    output_kp_path = "output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-dfbra_aug/model_best-kp.pth"
    output_limb_path = "output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-dfbra_aug/model_best-limb.pth"

    convert(checkpoint_path, output_kp_path, output_limb_path)
