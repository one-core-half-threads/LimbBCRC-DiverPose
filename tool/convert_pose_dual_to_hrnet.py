import argparse
from collections import OrderedDict
import torch


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")

    # checkpoint.pth.tar 通常是一个 dict，里面有 state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # final_state.pth.tar 通常可能直接就是 state_dict
        state_dict = checkpoint

    # 兼容 DataParallel / DDP 保存出来的 module. 前缀
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    return cleaned


def convert(src, dst, net):
    assert net in ["keypoint_net", "limb_net"], "net must be resnet or resnet2"

    state_dict = load_state_dict(src)
    prefix = net + "."

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v

    if len(new_state_dict) == 0:
        raise RuntimeError(
            f"No parameters with prefix '{prefix}' found in {src}. "
            f"Please check whether this is a PoseDual checkpoint."
        )

    torch.save(new_state_dict, dst)

    print(f"Saved converted HRNet weights to: {dst}")
    print(f"Extracted network: {net}")
    print(f"Number of tensors: {len(new_state_dict)}")

    for name in ["conv1.weight", "bn1.weight", "final_layer.weight", "final_layer.bias"]:
        if name in new_state_dict:
            print(name, tuple(new_state_dict[name].shape))
        else:
            print("Missing:", name)


if __name__ == "__main__":
    # PoseDual checkpoint path
    checkpoint_path = 'output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-dfbra_aug/model_best.pth.tar'
    # Output HRNet weight path
    output_path = 'output/mix_mpii_ai/pose_hetero_limb_48/w48kp_w48limb_256x256_HeteroLimb_MPII16_hetero_limb_w48-dfbra_aug/model_best-kp.pth'
    # which HRNet branch to extract: "resnet", "resnet2"      
    net = 'keypoint_net'

    convert(checkpoint_path, output_path, net)

