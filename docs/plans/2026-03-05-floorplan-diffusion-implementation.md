# Floorplan Diffusion Model — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce a constrained latent transformer-based diffusion model that generates accessible multi-occupancy floor plans at 512×512, with wandb live tracking and multi-GPU DDP training.

**Architecture:** Frozen SD 2.1 VAE encodes floor plans to 64×64×4 latent space. A trainable condition encoder maps 512×512 condition images to the same latent shape. A 28-block ViT denoises the concatenated (noisy latent + condition latent) to predict noise. DDPM with 1000 timesteps.

**Tech Stack:** PyTorch 2.x, diffusers (for VAE), wandb, Shapely (preprocessing), Pillow, torchvision, scipy, scikit-learn

---

### Task 1: Project Scaffolding + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `configs/train_config.yaml`
- Create: `src/__init__.py`
- Create: `data/__init__.py`

**Step 1: Create requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.25.0
transformers>=4.30.0
accelerate>=0.25.0
wandb>=0.16.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
shapely>=2.0.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
tqdm>=4.66.0
pyyaml>=6.0
pytorch-fid>=0.3.0
kaggle>=1.6.0
einops>=0.7.0
```

**Step 2: Create config file**

```yaml
# configs/train_config.yaml
project_name: floorplan-diffusion

# Data
data_dir: data/msd_processed
image_size: 512
latent_size: 64
latent_channels: 4

# Model
vae_model: stabilityai/sd-vae-ft-mse
condition_encoder:
  in_channels: 3
  out_channels: 4
  base_channels: 64
vit:
  input_channels: 8  # 4 (noisy latent) + 4 (condition)
  patch_size: 2
  embed_dim: 768
  depth: 28
  num_heads: 16
  mlp_ratio: 4.0

# Diffusion
num_timesteps: 1000
beta_start: 0.0001
beta_end: 0.02
beta_schedule: linear

# Training
batch_size: 8  # per GPU
epochs: 150
learning_rate: 0.0001
weight_decay: 0.01
num_workers: 4
seed: 42

# Checkpointing
checkpoint_dir: checkpoints
save_every: 10
sample_every: 5
num_samples: 8

# wandb
wandb_project: floorplan-diffusion
wandb_entity: null  # set to your wandb username
```

**Step 3: Create __init__.py files**

Empty files for `src/__init__.py` and `data/__init__.py`.

**Step 4: Commit**

```bash
git init
git add requirements.txt configs/ src/__init__.py data/__init__.py
git commit -m "feat: project scaffolding with dependencies and config"
```

---

### Task 2: Dataset Download + Preprocessing

**Files:**
- Create: `data/download_msd.py`
- Create: `data/preprocess.py`

**Step 1: Write download script**

`data/download_msd.py` — Downloads MSD dataset from Kaggle. User needs `~/.kaggle/kaggle.json` or env vars.

The script should:
1. Download from `kaggle datasets download -d caspervanengelenburg/modified-swiss-dwellings`
2. Extract to `data/msd_raw/`
3. Verify expected directory structure

**Step 2: Write preprocessing script**

`data/preprocess.py` — Processes raw MSD data into training pairs.

The script must:
1. Load each floor plan image (from `full_out/`) and its annotations
2. For each room in the floor plan:
   - Extract the room mask (from pixel-wise annotations using room type colors)
   - Compute the bounding box (minimum rotated rectangle via Shapely)
   - Compute the inscribed circle (largest circle fitting in the room polygon)
   - Randomly assign one condition type per room: mask, bbox, circle, or unconditioned
3. Build the 512×512 condition image:
   - Plot circles first (sorted by area, largest first)
   - Plot bounding boxes (sorted by area, largest first)
   - Plot room masks last (so they're not obscured)
   - Add global conditions: boundary and/or structural plan
4. Create 2 versions per floor plan: with and without global conditions
5. Apply augmentation: 90° and 180° rotations (3× the data)
6. Split into train (8544) / val (600) / test (1600) before augmentation
7. Save as pairs of (condition_image.png, floor_plan.png) in organized directories

**Room type color mapping** (must match MSD annotation colors — extract from dataset):

The MSD dataset uses specific RGB colors per room type. These must be extracted from the dataset documentation/code. The condition images use the same color coding.

**Output structure:**
```
data/msd_processed/
├── train/
│   ├── conditions/   # 512×512 condition images
│   └── floor_plans/  # 512×512 floor plan images
├── val/
│   ├── conditions/
│   └── floor_plans/
└── test/
    ├── conditions/
    └── floor_plans/
```

**Step 3: Test preprocessing on a few samples**

Run on 10 floor plans, visually inspect output condition images.

**Step 4: Run full preprocessing**

```bash
python data/download_msd.py
python data/preprocess.py --data_dir data/msd_raw --output_dir data/msd_processed
```

**Step 5: Commit**

```bash
git add data/download_msd.py data/preprocess.py
git commit -m "feat: MSD dataset download and preprocessing pipeline"
```

---

### Task 3: PyTorch Dataset

**Files:**
- Create: `src/dataset.py`

**Step 1: Implement FloorPlanDataset**

```python
class FloorPlanDataset(torch.utils.data.Dataset):
    """
    Loads paired (condition_image, floor_plan) from preprocessed directories.
    Both are 512×512×3 images, normalized to [-1, 1].
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.condition_dir = os.path.join(root_dir, split, 'conditions')
        self.floorplan_dir = os.path.join(root_dir, split, 'floor_plans')
        self.filenames = sorted(os.listdir(self.floorplan_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        floorplan = Image.open(os.path.join(self.floorplan_dir, fname)).convert('RGB')
        condition = Image.open(os.path.join(self.condition_dir, fname)).convert('RGB')

        # Resize to 512×512 if needed
        floorplan = floorplan.resize((512, 512), Image.BILINEAR)
        condition = condition.resize((512, 512), Image.BILINEAR)

        # To tensor and normalize to [-1, 1]
        floorplan = transforms.ToTensor()(floorplan) * 2 - 1
        condition = transforms.ToTensor()(condition) * 2 - 1

        return condition, floorplan
```

**Step 2: Quick test — load a batch, check shapes**

```python
ds = FloorPlanDataset('data/msd_processed', split='train')
cond, fp = ds[0]
assert cond.shape == (3, 512, 512)
assert fp.shape == (3, 512, 512)
assert -1 <= cond.min() and cond.max() <= 1
```

**Step 3: Commit**

```bash
git add src/dataset.py
git commit -m "feat: FloorPlanDataset for loading condition/floorplan pairs"
```

---

### Task 4: Condition Encoder

**Files:**
- Create: `src/condition_encoder.py`

**Step 1: Implement ConditionEncoder**

A convolutional network that takes 512×512×3 condition images and outputs 64×64×4 latent representations (matching VAE latent shape).

```python
class ConditionEncoder(nn.Module):
    """
    Encodes 512×512×3 condition image → 64×64×4 condition latent.
    Uses strided convolutions to downsample 8× (512 → 64).
    """
    def __init__(self, in_channels=3, out_channels=4, base_channels=64):
        super().__init__()
        # 512→256→128→64 with increasing channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),      # 512→256
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),  # 256→128
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1), # 128→64
            nn.GroupNorm(8, base_channels*4),
            nn.SiLU(),
            nn.Conv2d(base_channels*4, out_channels, 3, 1, 1),    # 64→64, project to latent channels
        )

    def forward(self, x):
        return self.encoder(x)
```

**Step 2: Test shape**

```python
enc = ConditionEncoder()
x = torch.randn(2, 3, 512, 512)
out = enc(x)
assert out.shape == (2, 4, 64, 64)
```

**Step 3: Commit**

```bash
git add src/condition_encoder.py
git commit -m "feat: condition encoder (512×512×3 → 64×64×4)"
```

---

### Task 5: Vision Transformer Denoiser

**Files:**
- Create: `src/vit_denoiser.py`

**Step 1: Implement ViT denoiser**

The ViT takes patchified (64×64×8) input (noisy latent concatenated with condition latent), plus timestep embedding, and predicts noise (64×64×4).

Key components:
- **Patch embedding**: Conv2d with kernel=patch_size, stride=patch_size. Patch size 2 → 32×32 = 1024 tokens.
- **Timestep embedding**: Sinusoidal embedding → MLP → added to each token
- **Transformer blocks** (×28): LayerNorm → Multi-head self-attention (16 heads) → residual → LayerNorm → Pointwise FFN → residual
- **Output head**: LayerNorm → Linear → reshape to (64×64×4)

```python
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP"""

class TransformerBlock(nn.Module):
    """LayerNorm → MHSA → residual → LayerNorm → FFN → residual"""

class ViTDenoiser(nn.Module):
    """
    Input: (B, 8, 64, 64) — concat of noisy latent + condition latent
    Timestep: (B,) — integer timesteps
    Output: (B, 4, 64, 64) — predicted noise
    """
    def __init__(self, in_channels=8, out_channels=4, patch_size=2,
                 embed_dim=768, depth=28, num_heads=16, mlp_ratio=4.0):
        # Patch embed: (B, 8, 64, 64) → (B, 1024, embed_dim)
        # + timestep embedding
        # 28 transformer blocks
        # Output projection: (B, 1024, embed_dim) → (B, 1024, patch_size*patch_size*out_channels)
        # Reshape → (B, out_channels, 64, 64)
```

**Step 2: Test shapes**

```python
model = ViTDenoiser()
x = torch.randn(2, 8, 64, 64)
t = torch.randint(0, 1000, (2,))
out = model(x, t)
assert out.shape == (2, 4, 64, 64)
```

**Step 3: Commit**

```bash
git add src/vit_denoiser.py
git commit -m "feat: ViT denoiser (28 blocks, 16 heads, patch size 2)"
```

---

### Task 6: DDPM Diffusion Module

**Files:**
- Create: `src/diffusion.py`

**Step 1: Implement GaussianDiffusion**

```python
class GaussianDiffusion:
    """
    DDPM forward/reverse process.
    - Linear β schedule from β₁=1e-4 to βT=0.02
    - T=1000 timesteps
    - Forward: q(xₜ|x₀) = N(xₜ; √ᾱₜ·x₀, (1-ᾱₜ)I)
    - Training: sample t, noise, compute xₜ, predict noise, MSE loss
    - Sampling: reverse process from xT ~ N(0,I)
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t"""

    def p_losses(self, model, x0, condition, t):
        """Training loss: MSE between predicted and true noise"""

    @torch.no_grad()
    def p_sample(self, model, xt, condition, t):
        """Single reverse step"""

    @torch.no_grad()
    def sample(self, model, condition, shape):
        """Full reverse process: generate from noise"""
```

**Step 2: Test forward/reverse shapes**

```python
diffusion = GaussianDiffusion()
x0 = torch.randn(2, 4, 64, 64)
t = torch.randint(0, 1000, (2,))
noisy = diffusion.q_sample(x0, t)
assert noisy.shape == x0.shape
```

**Step 3: Commit**

```bash
git add src/diffusion.py
git commit -m "feat: DDPM diffusion with linear beta schedule"
```

---

### Task 7: Full Model (VAE + Condition Encoder + ViT)

**Files:**
- Create: `src/model.py`

**Step 1: Implement FloorPlanDiffusionModel**

```python
class FloorPlanDiffusionModel(nn.Module):
    """
    Combines:
    - Frozen SD 2.1 VAE (encoder + decoder)
    - Trainable condition encoder
    - Trainable ViT denoiser

    Training forward:
        1. Encode floor_plan → x0 (via frozen VAE encoder)
        2. Encode condition → cond (via condition encoder)
        3. Return x0, cond for diffusion loss computation

    Inference:
        1. Encode condition → cond
        2. Sample via diffusion reverse process
        3. Decode latent → floor_plan (via frozen VAE decoder)
    """
    def __init__(self, config):
        super().__init__()
        # Load frozen VAE
        self.vae = AutoencoderKL.from_pretrained(config['vae_model'])
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.condition_encoder = ConditionEncoder(...)
        self.denoiser = ViTDenoiser(...)

    def encode_floorplan(self, x):
        """Encode 512×512 floor plan to 64×64×4 latent"""
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latent(self, z):
        """Decode 64×64×4 latent to 512×512 floor plan"""
        with torch.no_grad():
            z = z / self.vae.config.scaling_factor
            return self.vae.decode(z).sample

    def encode_condition(self, cond_img):
        """Encode 512×512 condition image to 64×64×4"""
        return self.condition_encoder(cond_img)
```

**Step 2: Test end-to-end shapes**

```python
model = FloorPlanDiffusionModel(config)
cond = torch.randn(2, 3, 512, 512)
fp = torch.randn(2, 3, 512, 512)
x0 = model.encode_floorplan(fp)
cond_latent = model.encode_condition(cond)
assert x0.shape == (2, 4, 64, 64)
assert cond_latent.shape == (2, 4, 64, 64)
```

**Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: full model combining frozen VAE + condition encoder + ViT"
```

---

### Task 8: Training Script with DDP + wandb

**Files:**
- Create: `train.py`

**Step 1: Implement training script**

Key features:
- **DDP setup**: `torchrun --nproc_per_node=4 train.py`
- **wandb init** on rank 0 only
- **Training loop**:
  1. Load batch (condition, floor_plan)
  2. Encode floor_plan → x0 (VAE, no grad)
  3. Encode condition → cond_latent (condition encoder, grad)
  4. Sample t ~ Uniform(0, T)
  5. Add noise: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
  6. Concat [xₜ, cond_latent] → ViT → predicted noise
  7. Loss = MSE(predicted, ε)
  8. Backward, optimizer step
- **Validation**: every epoch, compute val loss
- **Sampling**: every `sample_every` epochs, generate images and log to wandb
- **Checkpointing**: save every `save_every` epochs + best val loss
- **Logging**: loss, LR, GPU memory, epoch to wandb every step
- **Resume**: support `--resume checkpoint.pt` to continue interrupted training

```python
# Pseudo-structure:
def main():
    args = parse_args()
    config = load_config(args.config)

    # DDP setup
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Model
    model = FloorPlanDiffusionModel(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Diffusion
    diffusion = GaussianDiffusion(...)

    # Optimizer (only trainable params: condition encoder + ViT)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])

    # Dataset + DistributedSampler
    train_dataset = FloorPlanDataset(config['data_dir'], 'train')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=train_sampler, num_workers=config['num_workers'])

    # wandb (rank 0 only)
    if local_rank == 0:
        wandb.init(project=config['wandb_project'], config=config)

    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, diffusion, train_loader, optimizer, epoch, config, local_rank)
        val_loss = validate(model, diffusion, val_loader, local_rank)

        if local_rank == 0:
            wandb.log({'val_loss': val_loss, 'epoch': epoch})

            if epoch % config['sample_every'] == 0:
                generate_and_log_samples(model, diffusion, val_dataset, config)

            if epoch % config['save_every'] == 0:
                save_checkpoint(model, optimizer, epoch, val_loss)

            if val_loss < best_val_loss:
                save_checkpoint(model, optimizer, epoch, val_loss, is_best=True)
```

**Step 2: Test locally on dev cluster (1 GPU, 2 epochs, small subset)**

```bash
python train.py --config configs/train_config.yaml --debug --epochs 2
```

Verify: wandb logs appear, checkpoint saved, no OOM.

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: DDP training script with wandb logging and checkpointing"
```

---

### Task 9: Inference / Generation Script

**Files:**
- Create: `generate.py`

**Step 1: Implement generation script**

```python
"""
Usage:
    python generate.py --checkpoint best.pt --condition_dir test/conditions/ --output_dir outputs/
    python generate.py --checkpoint best.pt --condition_image my_condition.png --output output.png
"""
def generate(model, diffusion, condition_image):
    cond_latent = model.encode_condition(condition_image)
    shape = (1, 4, 64, 64)
    latent = diffusion.sample(model.denoiser, cond_latent, shape)
    image = model.decode_latent(latent)
    return image
```

Supports:
- Single condition image → single output
- Directory of conditions → directory of outputs
- Batch generation with progress bar
- Optional: DDIM sampling for faster inference (configurable number of steps)

**Step 2: Commit**

```bash
git add generate.py
git commit -m "feat: inference script for generating floor plans from conditions"
```

---

### Task 10: Evaluation Script (FID + MIoU)

**Files:**
- Create: `evaluate.py`

**Step 1: Implement evaluation**

- **FID**: Use `pytorch-fid` library to compute between generated and ground truth directories
- **MIoU**: K-means color segmentation of generated images, then IoU per room type vs ground truth masks

```python
"""
Usage:
    python evaluate.py --generated_dir outputs/ --ground_truth_dir data/msd_processed/test/floor_plans/
"""
```

**Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: FID and MIoU evaluation metrics"
```

---

### Task 11: Docker + Deployment

**Files:**
- Create: `Dockerfile`
- Create: `scripts/run_training.sh`

**Step 1: Dockerfile**

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# Python, pip, requirements
# Copy code
# Default command: training
```

**Step 2: Training launch script**

```bash
#!/bin/bash
# scripts/run_training.sh
# Sets up wandb, launches DDP training on 4 GPUs
export WANDB_API_KEY=${WANDB_API_KEY}
torchrun --nproc_per_node=4 --master_port=29500 \
    train.py --config configs/train_config.yaml
```

**Step 3: Commit**

```bash
git add Dockerfile scripts/
git commit -m "feat: Docker and training launch scripts for GCP"
```

---

### Task 12: README with Full Instructions

**Files:**
- Create: `README.md`

**Step 1: Write README covering:**

1. Setup (local + GCP)
2. Dataset download + preprocessing
3. Training (local debug + full GCP)
4. Monitoring via wandb
5. Inference
6. Evaluation

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, training, and evaluation instructions"
```

---

### Task 13: Local Testing on Dev Cluster

**Step 1: Run preprocessing on a tiny subset (10 floor plans)**
**Step 2: Train for 2-3 epochs with batch_size=1 on the RTX A5000**
**Step 3: Verify wandb logging works**
**Step 4: Verify checkpoint saving/loading**
**Step 5: Run inference on a sample**
**Step 6: Run evaluation pipeline**

If the A5000 has enough free VRAM (~17GB available), we may be able to run with batch_size=1 or 2 for testing. The model should fit in ~12-15GB.

---

## Execution Order

Tasks 1-7 can be developed on the dev cluster without GPU.
Task 8 needs GPU for testing.
Tasks 9-10 can be developed in parallel with 8.
Task 11-12 are final packaging.
Task 13 is integration testing.

## Parallelization Opportunities

- Tasks 4, 5, 6 are independent (condition encoder, ViT, diffusion module)
- Tasks 9, 10 are independent of each other
- Task 2 (preprocessing) is independent of tasks 3-7 (model code)
