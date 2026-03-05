# Floorplan Diffusion Model — Design Document

## Paper
"Generating accessible multi-occupancy floor plans with fine-grained control using a diffusion model"
Zhang & Zhang, Automation in Construction 177 (2025) 106332

## Goal
Reproduce the paper's constrained latent transformer-based diffusion model for generating
accessible multi-occupancy floor plans at 512×512 resolution conditioned on flexible design constraints.

## Decisions
- **VAE**: Stable Diffusion 2.1 (`stabilityai/sd-vae-ft-mse`), frozen during training
- **Backbone**: Custom ViT, 28 transformer blocks, 16-head attention
- **Latent space**: 64×64×4 (SD VAE output for 512×512 input)
- **Condition encoder**: Conv network, 512×512×3 → 64×64×4, trainable
- **Diffusion**: DDPM, 1000 timesteps, linear β schedule (β₁=1e-4, βT=0.02)
- **Training**: DDP 4×A100 80GB, effective batch 32 (8/GPU), AdamW LR 1e-4, 150 epochs
- **Dataset**: MSD (Modified Swiss Dwellings), 25,632 augmented training samples
- **Monitoring**: Weights & Biases (loss, LR, GPU stats, sample images every 5 epochs)
- **Checkpointing**: every 10 epochs + best val loss
- **Refinement/accessibility checker**: deferred to phase 2

## Architecture

### Condition Image (512×512×3)
Built from room-level + global-level constraints:
1. Sort circles + bounding boxes by area (largest first)
2. Plot circles → bounding boxes → room masks (masks on top)
3. Color-coded by 13 room type categories
4. Add global conditions (boundary, structural plan) last

### Model Components
1. **Frozen VAE Encoder**: 512×512×3 floor plan → 64×64×4 latent
2. **Condition Encoder**: 512×512×3 condition image → 64×64×4 condition latent (trainable conv net)
3. **ViT Denoiser**: Takes concatenated (xₜ + condition) = 64×64×8, patchified → 28 transformer blocks → predicts noise ε_θ
4. **Frozen VAE Decoder**: 64×64×4 latent → 512×512×3 floor plan

### Training Loop
1. Load (condition_image, floor_plan) pair
2. Encode floor_plan → x₀ via frozen VAE
3. Encode condition_image → cond via condition encoder
4. Sample t ~ Uniform(1, T), noise ε ~ N(0,I)
5. xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
6. Concatenate [xₜ, cond] → ViT → predicted ε_θ
7. Loss = MSE(ε_θ, ε)

### Inference
1. Build condition image from constraints
2. Encode condition → cond latent
3. xT ~ N(0,I)
4. For t = T..1: denoise with ViT conditioned on cond
5. Decode x₀ via VAE decoder → 512×512 floor plan

## Dataset: MSD
- 5,372 building floor plans, ~18,900 apartments
- 13 room type categories
- Preprocessing: extract room masks, bounding boxes, circles from annotations
- Split: 8,544 train / 600 val / 1,600 test (before augmentation)
- Augmentation: 90° and 180° rotations → 25,632 training samples
- Each sample has 2 versions: with and without global conditions

## Evaluation (post-training)
- **FID**: Fréchet Inception Distance (quality)
- **MIoU**: Mean Intersection over Union (constraint adherence)

## Project Structure
```
floorplan-diffusion/
├── configs/train_config.yaml
├── data/
│   ├── download_msd.py
│   └── preprocess.py
├── src/
│   ├── dataset.py
│   ├── condition_encoder.py
│   ├── vit_denoiser.py
│   ├── diffusion.py
│   ├── model.py
│   └── condition_image.py
├── train.py
├── generate.py
├── evaluate.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Room Type Color Map (from MSD dataset)
| Room Type   | Index |
|-------------|-------|
| Balcony     | 0     |
| Kitchen     | 1     |
| Bedroom     | 2     |
| Stairs      | 3     |
| Corridor    | 4     |
| Storeroom   | 5     |
| Bathroom    | 6     |
| Living room | 7     |
| Wall        | 8     |
| Outdoor     | 9     |
| Railing     | 10    |
| Background  | 11    |
| Door        | 12    |

## Phase 2 (later)
- Accessibility checker (ADA/IBC rules)
- Iterative refinement loop
- K-means color normalization for room segmentation
