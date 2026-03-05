# Floorplan Diffusion Model

Reproduction of "Generating accessible multi-occupancy floor plans with fine-grained control using a diffusion model" (Zhang & Zhang, Automation in Construction 2025).

A constrained latent transformer-based diffusion model that generates 512×512 multi-occupancy floor plans conditioned on flexible design constraints.

## Architecture

- **VAE**: Frozen Stable Diffusion 2.1 VAE (512×512 → 64×64×4 latent space)
- **Condition Encoder**: Trainable conv network (512×512×3 condition image → 64×64×4)
- **ViT Denoiser**: 28-block Vision Transformer, 16 heads, 768 embed dim (~200M params)
- **Diffusion**: DDPM with 1000 timesteps, linear β schedule

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv

source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download & Preprocess Dataset

```bash
# Option A: Kaggle CLI (needs ~/.kaggle/kaggle.json)
pip install kaggle
python data/download_msd.py --output_dir data/msd_raw

# Option B: Manual download from
# https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings
# Extract to data/msd_raw/

# Preprocess into condition/floorplan pairs
python data/preprocess.py --raw_dir data/msd_raw --output_dir data/msd_processed
```

### 3. Train

```bash
# Set wandb API key
export WANDB_API_KEY=your_key_here

# Debug mode (1 GPU, 2 epochs, small batch)
python train.py --config configs/train_config.yaml --debug

# Full training on 4 GPUs
torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/train_config.yaml

# Resume from checkpoint
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_0050.pt
```

### 4. Monitor Training

Go to [wandb.ai](https://wandb.ai) → your project. You'll see:
- **Loss curves** (train/val) updated every 10 steps
- **Sample generations** (condition | generated | ground truth) every 5 epochs
- **GPU memory/utilization** stats

### 5. Generate Floor Plans

```bash
# From a single condition image
python generate.py --checkpoint checkpoints/best_model.pt --condition_image input.png --output output.png

# From a directory of conditions
python generate.py --checkpoint checkpoints/best_model.pt --condition_dir data/msd_processed/test/conditions/ --output_dir outputs/

# Faster generation with fewer DDIM steps
python generate.py --checkpoint checkpoints/best_model.pt --condition_dir test_conditions/ --output_dir outputs/ --steps 25
```

### 6. Evaluate

```bash
# FID (quality) + MIoU (constraint adherence)
python evaluate.py --generated_dir outputs/ --ground_truth_dir data/msd_processed/test/floor_plans/ --metrics fid miou
```

## GCP Deployment

```bash
# On your GCP instance with 4× A100:
bash scripts/setup_gcp.sh

# Or with Docker:
docker build -t floorplan-diffusion .
docker run --gpus all -e WANDB_API_KEY=$WANDB_API_KEY \
  -v /path/to/data:/app/data \
  -v /path/to/checkpoints:/app/checkpoints \
  floorplan-diffusion
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MSD (Modified Swiss Dwellings) |
| Training samples | ~25,632 (with augmentation) |
| Batch size | 32 effective (8/GPU × 4 GPUs) |
| Optimizer | AdamW, LR=1e-4, weight decay=0.01 |
| Epochs | 150 |
| Mixed precision | FP16 |
| Gradient clipping | Max norm 1.0 |

## Project Structure

```
├── configs/train_config.yaml    # All hyperparameters
├── data/
│   ├── download_msd.py          # Dataset download
│   └── preprocess.py            # Condition image generation
├── src/
│   ├── condition_encoder.py     # 512×512→64×64 condition encoder
│   ├── vit_denoiser.py          # 28-block ViT noise predictor
│   ├── diffusion.py             # DDPM + DDIM sampling
│   ├── model.py                 # Full model (VAE + encoder + ViT)
│   └── dataset.py               # PyTorch dataset
├── train.py                     # DDP training with wandb
├── generate.py                  # Inference
├── evaluate.py                  # FID + MIoU
├── Dockerfile                   # GCP container
└── scripts/
    ├── run_training.sh          # Training launch script
    └── setup_gcp.sh             # GCP setup
```
