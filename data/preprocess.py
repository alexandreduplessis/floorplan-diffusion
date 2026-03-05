#!/usr/bin/env python3
"""
Preprocess the MSD dataset into condition/floor-plan image pairs.

This script:
1. Color-segments floor plan images to identify rooms
2. For each room, computes mask, bounding box, and inscribed circle
3. Randomly assigns each room a conditioning type (mask, bbox, circle, unconditioned)
4. Builds condition images combining room conditions + optional global structure
5. Creates two versions per floor plan (with/without global structure)
6. Splits into train/val/test and augments training data with 90/180 deg rotations

Usage:
    python data/preprocess.py --raw_dir data/msd_raw --output_dir data/msd_processed

Paper reference: Section 4.1.1 of "Generating Multi-Occupancy Floor Plans with
Latent Diffusion" (2025).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MSD room-type color map (RGB).
# These colors are derived from the MSD dataset conventions. The dataset uses
# pixel-wise semantic color annotations with 13 room types.
# Source: https://github.com/caspervanengelenburg/msd and paper figures.
# ---------------------------------------------------------------------------
ROOM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "background":  (255, 255, 255),
    "outdoor":     (200, 200, 200),
    "wall":        (0,   0,   0),
    "railing":     (100, 100, 100),
    "door":        (140, 80,  50),
    "stair":       (50,  130, 80),
    "balcony":     (120, 190, 80),
    "kitchen":     (240, 165, 60),
    "bedroom":     (65,  105, 190),
    "corridor":    (210, 180, 140),
    "storeroom":   (70,  160, 160),
    "bathroom":    (150, 100, 180),
    "living_room": (210, 105, 50),
}

# Structural / non-room colors that should not be treated as rooms
NON_ROOM_LABELS = {"background", "outdoor", "wall", "railing", "door"}

# Room labels eligible for conditioning
ROOM_LABELS = sorted(set(ROOM_COLORS.keys()) - NON_ROOM_LABELS)

# Build reverse lookup: RGB tuple -> label
COLOR_TO_LABEL: Dict[Tuple[int, int, int], str] = {v: k for k, v in ROOM_COLORS.items()}

# Conditioning types
COND_TYPES = ["mask", "bbox", "circle", "unconditioned"]

# Target image size
IMG_SIZE = 512

# Dataset split sizes (before augmentation)
SPLIT_TRAIN = 8544
SPLIT_VAL = 600
SPLIT_TEST = 1600


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mask_to_polygon(mask: np.ndarray) -> Optional[Polygon]:
    """Convert a binary mask to a Shapely Polygon (largest connected component)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

    if not polygons:
        return None

    # Return the largest polygon
    return max(polygons, key=lambda p: p.area)


def minimum_rotated_rectangle(poly: Polygon) -> np.ndarray:
    """Return the corners of the minimum rotated bounding rectangle as Nx2 array."""
    rect = poly.minimum_rotated_rectangle
    coords = np.array(rect.exterior.coords[:-1])  # 4 corners
    return coords


def largest_inscribed_circle(poly: Polygon, tolerance: float = 1.0) -> Tuple[float, float, float]:
    """
    Approximate the largest inscribed circle using the distance transform on a
    rasterized version of the polygon.

    Returns (cx, cy, radius).
    """
    minx, miny, maxx, maxy = poly.bounds
    w = int(maxx - minx) + 2
    h = int(maxy - miny) + 2

    if w < 3 or h < 3:
        centroid = poly.centroid
        return (centroid.x, centroid.y, 1.0)

    # Rasterize polygon into a small mask
    mask = np.zeros((h, w), dtype=np.uint8)
    ext_coords = np.array(poly.exterior.coords)
    shifted = ext_coords - np.array([minx, miny])
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)

    cx = max_loc[0] + minx
    cy = max_loc[1] + miny
    radius = max(max_val, 1.0)

    return (cx, cy, radius)


# ---------------------------------------------------------------------------
# Room extraction
# ---------------------------------------------------------------------------

def extract_rooms(
    image: np.ndarray, color_tolerance: int = 10
) -> List[Dict]:
    """
    Extract individual rooms from a floor plan image via color segmentation.

    Each room dict contains:
        - label: str (room type name)
        - color: (R, G, B)
        - mask: np.ndarray (H, W) binary mask
        - polygon: Shapely Polygon
        - area: float
    """
    h, w = image.shape[:2]
    rooms = []
    visited = np.zeros((h, w), dtype=bool)

    for label in ROOM_LABELS:
        color = np.array(ROOM_COLORS[label], dtype=np.uint8)

        # Create a mask for pixels matching this color (with tolerance)
        diff = np.abs(image.astype(np.int16) - color.astype(np.int16))
        color_mask = np.all(diff <= color_tolerance, axis=2).astype(np.uint8)

        # Exclude already-visited pixels (in case of color overlap with tolerance)
        color_mask[visited] = 0

        if color_mask.sum() < 50:  # skip tiny regions
            continue

        # Find connected components to separate individual rooms of the same type
        num_labels, labels = cv2.connectedComponents(color_mask)

        for comp_id in range(1, num_labels):
            comp_mask = (labels == comp_id).astype(np.uint8)
            area = comp_mask.sum()

            if area < 100:  # skip very small components (noise)
                continue

            poly = mask_to_polygon(comp_mask)
            if poly is None:
                continue

            visited[comp_mask > 0] = True
            rooms.append({
                "label": label,
                "color": ROOM_COLORS[label],
                "mask": comp_mask,
                "polygon": poly,
                "area": float(area),
            })

    return rooms


# ---------------------------------------------------------------------------
# Condition image building
# ---------------------------------------------------------------------------

def assign_condition_types(
    rooms: List[Dict], rng: random.Random
) -> List[Dict]:
    """Randomly assign a conditioning type to each room."""
    for room in rooms:
        room["cond_type"] = rng.choice(COND_TYPES)
    return rooms


def draw_filled_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, color: Tuple):
    """Draw a filled circle on a PIL ImageDraw."""
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=color,
    )


def draw_filled_rotated_rect(
    image: np.ndarray, corners: np.ndarray, color: Tuple
):
    """Draw a filled rotated rectangle on a numpy image using cv2."""
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color=(color[2], color[1], color[0]))  # BGR for cv2


def build_condition_image(
    rooms: List[Dict],
    structure_img: Optional[np.ndarray] = None,
    img_size: int = IMG_SIZE,
) -> np.ndarray:
    """
    Build a condition image from room conditions.

    Layering order (back to front):
        1. White background
        2. Global structure (boundary/walls from structure_in) if provided
        3. Circles (largest area first)
        4. Bounding boxes (largest area first)
        5. Room masks (so they are fully visible on top)

    Unconditioned rooms are not drawn.
    """
    # Start with white background
    cond = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    # Add global structure if provided
    if structure_img is not None:
        struct_resized = cv2.resize(structure_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        # Overlay structure: where structure is not white, replace background
        non_white = np.any(struct_resized != 255, axis=2)
        cond[non_white] = struct_resized[non_white]

    # Separate rooms by conditioning type
    circle_rooms = [r for r in rooms if r["cond_type"] == "circle"]
    bbox_rooms = [r for r in rooms if r["cond_type"] == "bbox"]
    mask_rooms = [r for r in rooms if r["cond_type"] == "mask"]

    # Sort by area (largest first) so smaller rooms appear on top
    circle_rooms.sort(key=lambda r: r["area"], reverse=True)
    bbox_rooms.sort(key=lambda r: r["area"], reverse=True)
    mask_rooms.sort(key=lambda r: r["area"], reverse=True)

    # Draw circles
    for room in circle_rooms:
        cx, cy, radius = largest_inscribed_circle(room["polygon"])
        color = room["color"]
        cv2.circle(
            cond,
            (int(round(cx)), int(round(cy))),
            int(round(radius)),
            (color[0], color[1], color[2]),
            thickness=-1,  # filled
        )

    # Draw bounding boxes (rotated rectangles)
    for room in bbox_rooms:
        corners = minimum_rotated_rectangle(room["polygon"])
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        color = room["color"]
        # cv2 uses BGR internally but we keep our image in RGB; fillPoly writes
        # directly to the array so we pass RGB since our array is RGB.
        cv2.fillPoly(cond, [pts], color=(int(color[0]), int(color[1]), int(color[2])))

    # Draw masks (on top)
    for room in mask_rooms:
        mask = room["mask"]
        color = room["color"]
        for c in range(3):
            cond[:, :, c] = np.where(mask > 0, color[c], cond[:, :, c])

    return cond


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_image_rgb(path: str, size: Optional[int] = None) -> np.ndarray:
    """Load an image as RGB numpy array, optionally resizing."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.NEAREST)
    return np.array(img)


def save_image(arr: np.ndarray, path: str):
    """Save a numpy RGB array as a PNG image."""
    Image.fromarray(arr).save(path)


def rotate_image(arr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 90 or 180 degrees."""
    if angle == 90:
        return np.rot90(arr, k=1)
    elif angle == 180:
        return np.rot90(arr, k=2)
    elif angle == 270:
        return np.rot90(arr, k=3)
    return arr


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def gather_samples(raw_dir: str) -> List[Dict]:
    """
    Gather all floor plan samples from the raw MSD dataset.

    Returns a list of dicts with keys:
        - floorplan_path: path to full_out image
        - structure_path: path to structure_in image (may not exist for test)
        - sample_id: identifier string
        - split_source: 'train' or 'test'
    """
    samples = []
    for split in ["train", "test"]:
        fp_dir = Path(raw_dir) / split / "full_out"
        struct_dir = Path(raw_dir) / split / "structure_in"

        if not fp_dir.is_dir():
            print(f"[WARNING] Directory not found: {fp_dir}")
            continue

        for fp_file in sorted(fp_dir.iterdir()):
            if fp_file.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue

            struct_file = struct_dir / fp_file.name
            samples.append({
                "floorplan_path": str(fp_file),
                "structure_path": str(struct_file) if struct_file.exists() else None,
                "sample_id": fp_file.stem,
                "split_source": split,
            })

    return samples


def process_sample(
    sample: Dict,
    rng: random.Random,
    img_size: int = IMG_SIZE,
    color_tolerance: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single sample into (condition, floorplan) pairs.

    Returns a list of (condition_image, floorplan_image) tuples.
    Two versions are produced:
        1. Without global structure conditions
        2. With global structure conditions (boundary from structure_in)
    """
    # Load floor plan image
    fp_img = load_image_rgb(sample["floorplan_path"], size=img_size)

    # Extract rooms
    rooms = extract_rooms(fp_img, color_tolerance=color_tolerance)

    if len(rooms) == 0:
        # If no rooms found, still return the floor plan with a blank condition
        blank_cond = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        return [(blank_cond, fp_img)]

    # Assign random conditioning types
    rooms = assign_condition_types(rooms, rng)

    results = []

    # Version 1: without global structure
    cond_no_struct = build_condition_image(rooms, structure_img=None, img_size=img_size)
    results.append((cond_no_struct, fp_img))

    # Version 2: with global structure (if available)
    if sample["structure_path"] is not None and os.path.exists(sample["structure_path"]):
        struct_img = load_image_rgb(sample["structure_path"], size=img_size)
        cond_with_struct = build_condition_image(rooms, structure_img=struct_img, img_size=img_size)
        results.append((cond_with_struct, fp_img))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MSD dataset into condition/floorplan pairs."
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/msd_raw",
        help="Path to the raw MSD dataset directory (default: data/msd_raw)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/msd_processed",
        help="Output directory for processed pairs (default: data/msd_processed)",
    )
    parser.add_argument(
        "--img_size", type=int, default=IMG_SIZE,
        help=f"Target image size (default: {IMG_SIZE})",
    )
    parser.add_argument(
        "--color_tolerance", type=int, default=10,
        help="Tolerance for color matching during segmentation (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n_train", type=int, default=SPLIT_TRAIN,
        help=f"Number of training samples before augmentation (default: {SPLIT_TRAIN})",
    )
    parser.add_argument(
        "--n_val", type=int, default=SPLIT_VAL,
        help=f"Number of validation samples (default: {SPLIT_VAL})",
    )
    parser.add_argument(
        "--n_test", type=int, default=SPLIT_TEST,
        help=f"Number of test samples (default: {SPLIT_TEST})",
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable rotation augmentation for training data.",
    )
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    output_dir = os.path.abspath(args.output_dir)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Step 1: Gather all samples
    # -----------------------------------------------------------------------
    print("[INFO] Scanning raw dataset ...")
    samples = gather_samples(raw_dir)
    print(f"[INFO] Found {len(samples)} floor plan images.")

    if len(samples) == 0:
        print("[ERROR] No samples found. Check --raw_dir path.")
        sys.exit(1)

    # Shuffle deterministically
    rng.shuffle(samples)

    # -----------------------------------------------------------------------
    # Step 2: Split into train / val / test
    # -----------------------------------------------------------------------
    total_needed = args.n_train + args.n_val + args.n_test
    if len(samples) < total_needed:
        print(
            f"[WARNING] Only {len(samples)} samples available, "
            f"but {total_needed} requested (train={args.n_train}, val={args.n_val}, test={args.n_test}). "
            f"Adjusting splits proportionally."
        )
        ratio = len(samples) / total_needed
        n_train = int(args.n_train * ratio)
        n_val = int(args.n_val * ratio)
        n_test = len(samples) - n_train - n_val
    else:
        n_train = args.n_train
        n_val = args.n_val
        n_test = args.n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:n_train + n_val + n_test]

    print(f"[INFO] Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # -----------------------------------------------------------------------
    # Step 3: Create output directories
    # -----------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "conditions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "floor_plans"), exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 4: Process each split
    # -----------------------------------------------------------------------
    def process_split(
        split_name: str,
        split_samples: List[Dict],
        augment: bool = False,
    ):
        """Process a dataset split and save condition/floorplan pairs."""
        cond_dir = os.path.join(output_dir, split_name, "conditions")
        fp_dir = os.path.join(output_dir, split_name, "floor_plans")
        idx = 0

        desc = f"Processing {split_name}"
        for sample in tqdm(split_samples, desc=desc, unit="img"):
            try:
                pairs = process_sample(
                    sample, rng,
                    img_size=args.img_size,
                    color_tolerance=args.color_tolerance,
                )
            except Exception as e:
                print(f"\n[WARNING] Failed to process {sample['sample_id']}: {e}")
                continue

            for cond_img, fp_img in pairs:
                # Save original
                fname = f"{idx:05d}.png"
                save_image(cond_img, os.path.join(cond_dir, fname))
                save_image(fp_img, os.path.join(fp_dir, fname))
                idx += 1

                # Augmentation: 90 and 180 degree rotations
                if augment:
                    for angle in [90, 180]:
                        fname_aug = f"{idx:05d}.png"
                        cond_rot = rotate_image(cond_img, angle)
                        fp_rot = rotate_image(fp_img, angle)
                        save_image(cond_rot, os.path.join(cond_dir, fname_aug))
                        save_image(fp_rot, os.path.join(fp_dir, fname_aug))
                        idx += 1

        print(f"[INFO] {split_name}: saved {idx} pairs.")
        return idx

    # Process train (with augmentation), val, test
    augment_train = not args.no_augment
    n_train_total = process_split("train", train_samples, augment=augment_train)
    n_val_total = process_split("val", val_samples, augment=False)
    n_test_total = process_split("test", test_samples, augment=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Train: {n_train_total} pairs" + (" (with 90/180 deg augmentation)" if augment_train else ""))
    print(f"  Val:   {n_val_total} pairs")
    print(f"  Test:  {n_test_total} pairs")
    print(f"  Output: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
