# Semi-Supervised 2D Pose Estimation for Underwater Divers

## Python Environment

The training code uses the following Python environment:

```text
Python 3.9
PyTorch 2.4.1
```

## Training

Run the following command from the project root directory:

```bash
python pose_estimation/train.py \
  --cfg experiments/mix_diver/hrnet/w48kp_w48limb_256x256_HeteroLimb_MPII16.yaml \
  --gpus 0,1,2,3 \
  --workers 24 \
  --batch_size 12 \
  --epochs 100 \
  --lr 0.0001 \
  --lr_steps 30 70 \
  --seed 2026 \
  --repeat_aug \
  --repeat_times 1 \
  --jc_aug
```

## Evaluation

Run the following command from the project root directory to evaluate `model_best.pth.tar` and save visualization results:

```bash
python "pose_estimation/evaluate_model_best_vis(Ours).py" \
  --cfg experiments/mix_diver/hrnet/w48kp_w48limb_256x256_HeteroLimb_MPII16.yaml \
  --model-file output/model_best.pth.tar \
  --gpus 0 \
  --vis-dir output/vis_results
```

