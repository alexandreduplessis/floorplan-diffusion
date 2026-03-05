#!/usr/bin/env python3
"""
Inference script for generating floor plans from condition images
using a trained latent diffusion model.

Usage examples:
    # Single image
    python generate.py --checkpoint checkpoints/best_model.pt --condition_image input.png --output output.png

    # Directory of images
    python generate.py --checkpoint checkpoints/best_model.pt --condition_dir test/conditions/ --output_dir outputs/

    # DDIM with custom steps
    python generate.py --checkpoint checkpoints/best_model.pt --condition_dir test/conditions/ --output_dir outputs/ --steps 50

    # Full DDPM sampling
    python generate.py --checkpoint checkpoints/best_model.pt --condition_dir test/conditions/ --output_dir outputs/ --sampler ddpm
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.diffusion import GaussianDiffusion
from src.model import FloorPlanDiffusionModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate floor plans from condition images using a trained diffusion model."
    )

    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file).",
    )

    # Input: single image or directory (at least one required)
    parser.add_argument(
        "--condition_image", type=str, default=None,
        help="Path to a single condition image.",
    )
    parser.add_argument(
        "--condition_dir", type=str, default=None,
        help="Path to a directory of condition images.",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="output.png",
        help="Output path for single-image mode (default: output.png).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for directory mode (default: outputs/).",
    )

    # Sampling
    parser.add_argument(
        "--sampler", type=str, default="ddim", choices=["ddim", "ddpm"],
        help="Sampling method: ddim (fast, default) or ddpm (slow, higher quality).",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of DDIM inference steps (ignored for ddpm). Default: 50.",
    )
    parser.add_argument(
        "--eta", type=float, default=0.0,
        help="DDIM stochasticity parameter (0=deterministic, 1=DDPM-like). Default: 0.0.",
    )

    # Batching / device
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for generation (default: 4).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect cuda/cpu).",
    )

    # Config override
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to training config YAML (default: configs/train_config.yaml).",
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.condition_image is None and args.condition_dir is None:
        parser.error("At least one of --condition_image or --condition_dir is required.")

    return args


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_diffusion(checkpoint_path, config, device):
    """Load the model from a checkpoint and create the diffusion scheduler."""
    model = FloorPlanDiffusionModel(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Support checkpoints that store just state_dict or a dict with 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel / DistributedDataParallel prefixes
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned[key] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=config.get("num_timesteps", 1000),
        beta_start=config.get("beta_start", 1e-4),
        beta_end=config.get("beta_end", 0.02),
    )

    return model, diffusion


def build_transform(image_size=512):
    """Build the preprocessing transform: resize to 512x512, normalize to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
    ])


def collect_image_paths(condition_image=None, condition_dir=None):
    """Collect all condition image paths from arguments."""
    paths = []
    if condition_image is not None:
        p = Path(condition_image)
        if not p.is_file():
            print(f"Error: condition image not found: {p}", file=sys.stderr)
            sys.exit(1)
        paths.append(p)
    if condition_dir is not None:
        d = Path(condition_dir)
        if not d.is_dir():
            print(f"Error: condition directory not found: {d}", file=sys.stderr)
            sys.exit(1)
        for ext in sorted(IMAGE_EXTENSIONS):
            paths.extend(sorted(d.glob(f"*{ext}")))
            paths.extend(sorted(d.glob(f"*{ext.upper()}")))
    if not paths:
        print("Error: no condition images found.", file=sys.stderr)
        sys.exit(1)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def denormalize_and_save(tensor, path):
    """
    Convert a (3, H, W) tensor from [-1, 1] to [0, 255] uint8 and save as PNG.
    """
    img = tensor.clamp(-1, 1)
    img = (img + 1.0) / 2.0  # [0, 1]
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    Image.fromarray(img).save(path)


@torch.no_grad()
def generate(model, diffusion, image_paths, transform, args, device):
    """Generate floor plans for all condition images."""
    latent_shape_suffix = (
        4,
        args.config_data.get("latent_size", 64),
        args.config_data.get("latent_size", 64),
    )

    single_mode = args.condition_image is not None and args.condition_dir is None

    if not single_mode:
        os.makedirs(args.output_dir, exist_ok=True)

    total = len(image_paths)
    processed = 0

    pbar = tqdm(total=total, desc="Generating floor plans")

    while processed < total:
        batch_paths = image_paths[processed : processed + args.batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(transform(img))

        cond_batch = torch.stack(batch_imgs, dim=0).to(device)  # (B, 3, 512, 512)
        B = cond_batch.shape[0]

        # Encode conditions
        cond_latent = model.encode_condition(cond_batch)  # (B, 4, 64, 64)

        # Sample
        shape = (B, *latent_shape_suffix)
        if args.sampler == "ddim":
            z = diffusion.ddim_sample(
                model.denoiser, cond_latent, shape, device=device,
                num_inference_steps=args.steps, eta=args.eta,
            )
        else:
            z = diffusion.sample(model.denoiser, cond_latent, shape, device=device)

        # Decode
        decoded = model.decode_latent(z)  # (B, 3, 512, 512)

        # Save
        for i, p in enumerate(batch_paths):
            if single_mode:
                out_path = args.output
            else:
                out_path = os.path.join(args.output_dir, f"{p.stem}_generated.png")
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            denormalize_and_save(decoded[i], out_path)

        processed += B
        pbar.update(B)

    pbar.close()


def main():
    args = parse_args()

    # Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # Config
    config = load_config(args.config)
    args.config_data = config  # attach for use in generate()

    # Model + diffusion
    print(f"Loading checkpoint: {args.checkpoint}")
    model, diffusion = load_model_and_diffusion(args.checkpoint, config, device)
    print("Model loaded successfully.")

    # Collect images
    image_paths = collect_image_paths(args.condition_image, args.condition_dir)
    print(f"Found {len(image_paths)} condition image(s).")

    # Transform
    image_size = config.get("image_size", 512)
    transform = build_transform(image_size)

    # Generate
    generate(model, diffusion, image_paths, transform, args, device)

    single_mode = args.condition_image is not None and args.condition_dir is None
    if single_mode:
        print(f"Output saved to: {args.output}")
    else:
        print(f"Outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
