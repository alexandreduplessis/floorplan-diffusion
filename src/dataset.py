"""
PyTorch Dataset for preprocessed floor plan condition/target pairs.

Loads paired (condition_image, floor_plan) from the preprocessed directory
structure produced by data/preprocess.py. Both images are 512x512x3 RGB,
normalized to [-1, 1] for use with a latent diffusion model.

Usage:
    from src.dataset import FloorPlanDataset
    dataset = FloorPlanDataset("data/msd_processed", split="train")
    condition, floorplan = dataset[0]  # each is a [3, 512, 512] tensor in [-1, 1]
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FloorPlanDataset(Dataset):
    """
    Loads paired (condition_image, floor_plan) from preprocessed directories.
    Both are 512x512x3 images, normalized to [-1, 1].

    Directory structure expected:
        root_dir/
        ├── train/
        │   ├── conditions/
        │   │   ├── 00000.png
        │   │   └── ...
        │   └── floor_plans/
        │       ├── 00000.png
        │       └── ...
        ├── val/
        │   ├── conditions/
        │   └── floor_plans/
        └── test/
            ├── conditions/
            └── floor_plans/
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 512):
        """
        Args:
            root_dir: Path to the preprocessed dataset root (e.g., data/msd_processed).
            split: One of 'train', 'val', or 'test'.
            image_size: Target image size (default 512).
        """
        self.condition_dir = os.path.join(root_dir, split, "conditions")
        self.floorplan_dir = os.path.join(root_dir, split, "floor_plans")

        if not os.path.isdir(self.floorplan_dir):
            raise FileNotFoundError(
                f"Floor plan directory not found: {self.floorplan_dir}. "
                f"Run data/preprocess.py first."
            )
        if not os.path.isdir(self.condition_dir):
            raise FileNotFoundError(
                f"Condition directory not found: {self.condition_dir}. "
                f"Run data/preprocess.py first."
            )

        self.filenames = sorted(
            f for f in os.listdir(self.floorplan_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No images found in {self.floorplan_dir}. "
                f"Run data/preprocess.py first."
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                                    # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.filenames[idx]

        floorplan = Image.open(
            os.path.join(self.floorplan_dir, fname)
        ).convert("RGB")
        condition = Image.open(
            os.path.join(self.condition_dir, fname)
        ).convert("RGB")

        floorplan = self.transform(floorplan)
        condition = self.transform(condition)

        return condition, floorplan
