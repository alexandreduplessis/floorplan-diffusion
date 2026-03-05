#!/usr/bin/env python3
"""
Preprocess the MSD dataset into condition/floor-plan image pairs.

The MSD dataset stores floor plans as polygon geometries in a CSV file.
This script:
1. Renders floor plan images from polygon data (512×512 RGB)
2. For each room, computes bounding box and inscribed circle
3. Randomly assigns each room a conditioning type (mask, bbox, circle, unconditioned)
4. Builds condition images combining room conditions + optional structure
5. Creates two versions per floor plan (with/without structure)
6. Splits into train/val/test and augments training data with 90/180 deg rotations

Usage:
    python data/preprocess.py --csv_path data/msd_sample/mds_V2_5.372k.csv --output_dir data/msd_processed

    # Quick test with 10 floor plans:
    python data/preprocess.py --csv_path data/msd_sample/mds_V2_5.372k.csv --output_dir data/msd_processed --max_plans 10

Paper reference: Section 4.1.1 of "Generating accessible multi-occupancy
floor plans with fine-grained control using a diffusion model" (2025).
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MSD room-type color map (RGB) — from constants.py in the MSD repo.
# Order matches ROOM_NAMES in the MSD codebase.
# ---------------------------------------------------------------------------
ROOM_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Bedroom":       (31, 119, 180),   # #1f77b4
    "Livingroom":    (230, 85, 13),    # #e6550d
    "Kitchen":       (253, 141, 60),   # #fd8d3c
    "Dining":        (253, 174, 107),  # #fdae6b
    "Corridor":      (253, 208, 162),  # #fdd0a2
    "Stairs":        (114, 36, 108),   # #72246c
    "Storeroom":     (82, 84, 163),    # #5254a3
    "Bathroom":      (107, 110, 207),  # #6b6ecf
    "Balcony":       (44, 160, 44),    # #2ca02c
    "Structure":     (0, 0, 0),        # #000000
    "Door":          (255, 192, 0),    # #ffc000
    "Entrance Door": (152, 223, 138),  # #98df8a
    "Window":        (214, 39, 40),    # #d62728
}

# Room types that appear in condition images (not structural elements)
CONDITIONABLE_ROOMS = {
    "Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor",
    "Stairs", "Storeroom", "Bathroom", "Balcony",
}

# Structural elements (drawn as structure in floor plan, optionally in condition)
STRUCTURAL_TYPES = {"Structure", "Door", "Entrance Door", "Window"}

# Conditioning types for each room
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

def polygon_to_pixel_coords(poly, minx, miny, scale, img_size):
    """Convert a Shapely polygon's exterior coords to pixel coordinates."""
    coords = np.array(poly.exterior.coords)
    px = ((coords[:, 0] - minx) * scale).astype(np.int32)
    py = (img_size - 1 - (coords[:, 1] - miny) * scale).astype(np.int32)  # flip Y
    px = np.clip(px, 0, img_size - 1)
    py = np.clip(py, 0, img_size - 1)
    return np.stack([px, py], axis=1)


def largest_inscribed_circle(mask: np.ndarray) -> Tuple[int, int, float]:
    """Find the largest inscribed circle center and radius from a binary mask."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    return (max_loc[0], max_loc[1], max(max_val, 1.0))


def minimum_rotated_rect_pixels(mask: np.ndarray) -> Optional[np.ndarray]:
    """Get the minimum rotated rectangle corners from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    return box


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_floor_plan(plan_df, img_size=IMG_SIZE):
    """
    Render a floor plan from its polygon data to a 512×512 RGB image.

    Args:
        plan_df: DataFrame rows for one plan_id
        img_size: target image size

    Returns:
        image: (H, W, 3) uint8 RGB image
        rooms: list of dicts with {label, color, mask, area, polygon_pixels}
        structure_image: (H, W, 3) uint8 RGB image of structural elements only
        minx, miny, scale: transform parameters
    """
    # Parse all geometries
    entities = []
    for _, row in plan_df.iterrows():
        try:
            geom = wkt.loads(row["geom"])
        except Exception:
            continue
        room_type = row["roomtype"]
        if room_type not in ROOM_COLORS:
            continue
        entities.append({"geom": geom, "room_type": room_type})

    if not entities:
        return None, [], None, 0, 0, 1

    # Compute bounding box of the entire floor plan
    all_geoms = [e["geom"] for e in entities]
    union = unary_union(all_geoms)
    minx, miny, maxx, maxy = union.bounds

    # Add small padding
    pad = max(maxx - minx, maxy - miny) * 0.02
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    # Scale to fit in img_size
    span_x = maxx - minx
    span_y = maxy - miny
    scale = (img_size - 1) / max(span_x, span_y)

    # Render full floor plan image
    fp_image = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    structure_image = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    rooms = []

    # Draw structural elements first (background layer)
    for entity in entities:
        if entity["room_type"] in STRUCTURAL_TYPES:
            color = ROOM_COLORS[entity["room_type"]]
            geom = entity["geom"]
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
            for poly in polys:
                if not poly.is_valid or poly.is_empty:
                    continue
                pts = polygon_to_pixel_coords(poly, minx, miny, scale, img_size)
                cv2.fillPoly(fp_image, [pts], color)
                cv2.fillPoly(structure_image, [pts], color)

    # Draw room areas on top
    for entity in entities:
        if entity["room_type"] in CONDITIONABLE_ROOMS:
            color = ROOM_COLORS[entity["room_type"]]
            geom = entity["geom"]
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms) if geom.geom_type == "MultiPolygon" else []
            for poly in polys:
                if not poly.is_valid or poly.is_empty:
                    continue
                pts = polygon_to_pixel_coords(poly, minx, miny, scale, img_size)

                # Create mask for this room
                mask = np.zeros((img_size, img_size), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                area = mask.sum() / 255

                if area < 50:  # skip tiny rooms
                    continue

                cv2.fillPoly(fp_image, [pts], color)

                rooms.append({
                    "label": entity["room_type"],
                    "color": color,
                    "mask": mask,
                    "area": float(area),
                })

    return fp_image, rooms, structure_image, minx, miny, scale


def build_condition_image(rooms, structure_img=None, img_size=IMG_SIZE):
    """
    Build a condition image from room conditions.

    Layering order (back to front):
        1. White background
        2. Global structure (boundary/walls) if provided
        3. Circles (largest area first)
        4. Bounding boxes (largest area first)
        5. Room masks (so they are fully visible on top)

    Unconditioned rooms are not drawn.
    """
    cond = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    # Add global structure if provided
    if structure_img is not None:
        non_white = np.any(structure_img != 255, axis=2)
        cond[non_white] = structure_img[non_white]

    # Separate rooms by conditioning type
    circle_rooms = [r for r in rooms if r.get("cond_type") == "circle"]
    bbox_rooms = [r for r in rooms if r.get("cond_type") == "bbox"]
    mask_rooms = [r for r in rooms if r.get("cond_type") == "mask"]

    # Sort by area (largest first)
    circle_rooms.sort(key=lambda r: r["area"], reverse=True)
    bbox_rooms.sort(key=lambda r: r["area"], reverse=True)
    mask_rooms.sort(key=lambda r: r["area"], reverse=True)

    # Draw circles
    for room in circle_rooms:
        cx, cy, radius = largest_inscribed_circle(room["mask"])
        color = room["color"]
        cv2.circle(cond, (int(cx), int(cy)), int(radius), color, thickness=-1)

    # Draw bounding boxes (rotated rectangles)
    for room in bbox_rooms:
        box = minimum_rotated_rect_pixels(room["mask"])
        if box is not None:
            color = room["color"]
            cv2.fillPoly(cond, [box], color)

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

def save_image(arr: np.ndarray, path: str):
    """Save a numpy RGB array as a PNG image."""
    Image.fromarray(arr).save(path)


def rotate_image(arr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 90 or 180 degrees."""
    if angle == 90:
        return np.rot90(arr, k=1)
    elif angle == 180:
        return np.rot90(arr, k=2)
    return arr


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MSD dataset into condition/floorplan pairs."
    )
    parser.add_argument(
        "--csv_path", type=str, default="data/msd_sample/mds_V2_5.372k.csv",
        help="Path to MSD CSV file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/msd_processed",
        help="Output directory for processed pairs",
    )
    parser.add_argument(
        "--img_size", type=int, default=IMG_SIZE,
        help=f"Target image size (default: {IMG_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max_plans", type=int, default=None,
        help="Max number of plans to process (for testing)",
    )
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable rotation augmentation",
    )
    args = parser.parse_args()

    import pandas as pd

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # Step 1: Load CSV
    # -----------------------------------------------------------------------
    print(f"[INFO] Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    plan_ids = df["plan_id"].unique().tolist()
    rng.shuffle(plan_ids)
    print(f"[INFO] Found {len(plan_ids)} unique floor plans.")

    if args.max_plans:
        plan_ids = plan_ids[:args.max_plans]
        print(f"[INFO] Limiting to {len(plan_ids)} plans.")

    # -----------------------------------------------------------------------
    # Step 2: Split
    # -----------------------------------------------------------------------
    n_total = len(plan_ids)
    if n_total >= SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST:
        n_train, n_val, n_test = SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
    else:
        # Proportional split for small datasets
        ratio_train = SPLIT_TRAIN / (SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST)
        ratio_val = SPLIT_VAL / (SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST)
        n_train = int(n_total * ratio_train)
        n_val = int(n_total * ratio_val)
        n_test = n_total - n_train - n_val

    splits = {
        "train": plan_ids[:n_train],
        "val": plan_ids[n_train:n_train + n_val],
        "test": plan_ids[n_train + n_val:n_train + n_val + n_test],
    }
    print(f"[INFO] Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # -----------------------------------------------------------------------
    # Step 3: Create output directories
    # -----------------------------------------------------------------------
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split, "conditions"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, "floor_plans"), exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 4: Process each split
    # -----------------------------------------------------------------------
    def process_split(split_name, split_plan_ids, augment=False):
        cond_dir = os.path.join(args.output_dir, split_name, "conditions")
        fp_dir = os.path.join(args.output_dir, split_name, "floor_plans")
        idx = 0

        for plan_id in tqdm(split_plan_ids, desc=f"Processing {split_name}", unit="plan"):
            plan_df = df[df["plan_id"] == plan_id]

            try:
                fp_img, rooms, struct_img, _, _, _ = render_floor_plan(plan_df, args.img_size)
            except Exception as e:
                print(f"\n[WARNING] Failed to render plan {plan_id}: {e}")
                continue

            if fp_img is None or len(rooms) == 0:
                continue

            # Assign random condition types
            for room in rooms:
                room["cond_type"] = rng.choice(COND_TYPES)

            # Version 1: without structure
            cond_no_struct = build_condition_image(rooms, structure_img=None, img_size=args.img_size)
            save_image(cond_no_struct, os.path.join(cond_dir, f"{idx:05d}.png"))
            save_image(fp_img, os.path.join(fp_dir, f"{idx:05d}.png"))
            idx += 1

            # Version 2: with structure
            cond_with_struct = build_condition_image(rooms, structure_img=struct_img, img_size=args.img_size)
            save_image(cond_with_struct, os.path.join(cond_dir, f"{idx:05d}.png"))
            save_image(fp_img, os.path.join(fp_dir, f"{idx:05d}.png"))
            idx += 1

            # Augmentation
            if augment:
                for angle in [90, 180]:
                    for cond_img in [cond_no_struct, cond_with_struct]:
                        cond_rot = rotate_image(cond_img, angle)
                        fp_rot = rotate_image(fp_img, angle)
                        save_image(cond_rot, os.path.join(cond_dir, f"{idx:05d}.png"))
                        save_image(fp_rot, os.path.join(fp_dir, f"{idx:05d}.png"))
                        idx += 1

        print(f"[INFO] {split_name}: saved {idx} pairs.")
        return idx

    augment_train = not args.no_augment
    n_train_total = process_split("train", splits["train"], augment=augment_train)
    n_val_total = process_split("val", splits["val"], augment=False)
    n_test_total = process_split("test", splits["test"], augment=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Train: {n_train_total} pairs" + (" (with augmentation)" if augment_train else ""))
    print(f"  Val:   {n_val_total} pairs")
    print(f"  Test:  {n_test_total} pairs")
    print(f"  Output: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
