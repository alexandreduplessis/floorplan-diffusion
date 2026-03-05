"""
Evaluation metrics for generated floor plans: FID (quality) and MIoU (constraint adherence).

Usage:
    python evaluate.py --generated_dir outputs/ --ground_truth_dir data/msd_processed/test/floor_plans/ --metrics fid miou
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Room type colors (RGB) — from MSD constants.py
ROOM_COLORS = {
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
    "Background":    (255, 255, 255),
}

# Build lookup array: shape (N, 3) for vectorized nearest-color matching
ROOM_NAMES = list(ROOM_COLORS.keys())
ROOM_COLOR_ARRAY = np.array([ROOM_COLORS[name] for name in ROOM_NAMES], dtype=np.float32)


def classify_pixels_by_color(image_array: np.ndarray) -> np.ndarray:
    """Classify each pixel to the nearest room type color.

    Args:
        image_array: HxWx3 uint8 RGB image.

    Returns:
        HxW int array where each value is the index into ROOM_NAMES.
    """
    h, w, _ = image_array.shape
    pixels = image_array.reshape(-1, 3).astype(np.float32)  # (H*W, 3)
    # Compute squared distances to each room color: (H*W, N)
    diffs = pixels[:, None, :] - ROOM_COLOR_ARRAY[None, :, :]  # (H*W, N, 3)
    dists = np.sum(diffs ** 2, axis=2)  # (H*W, N)
    labels = np.argmin(dists, axis=1)  # (H*W,)
    return labels.reshape(h, w)


def compute_iou_per_class(pred_labels: np.ndarray, gt_labels: np.ndarray, num_classes: int):
    """Compute IoU for each class.

    Returns:
        dict mapping class index -> IoU (only for classes present in gt or pred).
    """
    ious = {}
    for c in range(num_classes):
        pred_mask = pred_labels == c
        gt_mask = gt_labels == c
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union > 0:
            ious[c] = intersection / union
    return ious


def compute_miou_color_matching(generated_dir: str, ground_truth_dir: str):
    """Compute MIoU using simple nearest-color per-pixel matching.

    For each image pair, classify pixels by nearest room color in both
    generated and ground truth, then compute IoU per room type.
    """
    gen_path = Path(generated_dir)
    gt_path = Path(ground_truth_dir)

    gen_files = sorted([f for f in gen_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    gt_files = sorted([f for f in gt_path.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])

    # Match files by name stem or by index
    if len(gen_files) == 0:
        print("No generated images found.")
        return {}
    if len(gt_files) == 0:
        print("No ground truth images found.")
        return {}

    # Try matching by filename stem
    gt_by_stem = {f.stem: f for f in gt_files}
    paired = []
    for gf in gen_files:
        if gf.stem in gt_by_stem:
            paired.append((gf, gt_by_stem[gf.stem]))

    # If no name matches, pair by sorted order (use min length)
    if len(paired) == 0:
        n = min(len(gen_files), len(gt_files))
        paired = list(zip(gen_files[:n], gt_files[:n]))
        print(f"No filename matches found; pairing by sorted order ({n} pairs).")
    else:
        print(f"Matched {len(paired)} image pairs by filename.")

    num_classes = len(ROOM_NAMES)
    # Accumulate per-class intersection and union across all images
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    per_image_ious = []

    for gen_file, gt_file in tqdm(paired, desc="Computing MIoU"):
        gen_img = np.array(Image.open(gen_file).convert("RGB"))
        gt_img = np.array(Image.open(gt_file).convert("RGB"))

        # Resize generated to match ground truth if needed
        if gen_img.shape[:2] != gt_img.shape[:2]:
            gen_pil = Image.fromarray(gen_img).resize(
                (gt_img.shape[1], gt_img.shape[0]), Image.NEAREST
            )
            gen_img = np.array(gen_pil)

        pred_labels = classify_pixels_by_color(gen_img)
        gt_labels = classify_pixels_by_color(gt_img)

        img_ious = compute_iou_per_class(pred_labels, gt_labels, num_classes)
        per_image_ious.append(img_ious)

        for c in range(num_classes):
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c
            total_intersection[c] += np.logical_and(pred_mask, gt_mask).sum()
            total_union[c] += np.logical_or(pred_mask, gt_mask).sum()

    # Compute global per-class IoU
    class_ious = {}
    for c in range(num_classes):
        if total_union[c] > 0:
            class_ious[ROOM_NAMES[c]] = total_intersection[c] / total_union[c]

    # Compute mean IoU (excluding classes with zero union)
    if class_ious:
        miou = np.mean(list(class_ious.values()))
    else:
        miou = 0.0

    return class_ious, miou, len(paired)


def compute_fid(generated_dir: str, ground_truth_dir: str, batch_size: int = 50, device: str = "cuda"):
    """Compute FID between generated and ground truth directories using pytorch_fid."""
    from pytorch_fid import fid_score

    fid_value = fid_score.calculate_fid_given_paths(
        [generated_dir, ground_truth_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )
    return fid_value


def print_table(title: str, rows: list, headers: list):
    """Print a simple formatted table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    print(f"\n{'=' * len(sep)}")
    print(f" {title}")
    print(f"{'=' * len(sep)}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        line = "|" + "|".join(f" {str(v):<{col_widths[i]}} " for i, v in enumerate(row)) + "|"
        print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated floor plans with FID and MIoU.")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory of generated images.")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Directory of ground truth images.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["fid", "miou"],
        default=["fid", "miou"],
        help="Which metrics to compute (default: both).",
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for FID computation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for FID computation (default: cuda).",
    )
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.generated_dir):
        print(f"Error: generated_dir '{args.generated_dir}' does not exist.")
        return
    if not os.path.isdir(args.ground_truth_dir):
        print(f"Error: ground_truth_dir '{args.ground_truth_dir}' does not exist.")
        return

    results = {}

    if "fid" in args.metrics:
        print("\n--- Computing FID ---")
        try:
            fid_value = compute_fid(
                args.generated_dir,
                args.ground_truth_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
            results["fid"] = fid_value
            print_table("FID Score", [["FID", f"{fid_value:.4f}"]], ["Metric", "Value"])
        except Exception as e:
            print(f"FID computation failed: {e}")

    if "miou" in args.metrics:
        print("\n--- Computing MIoU (Color Matching) ---")
        try:
            class_ious, miou, num_pairs = compute_miou_color_matching(
                args.generated_dir, args.ground_truth_dir
            )
            results["miou"] = miou
            results["class_ious"] = class_ious

            rows = []
            for name in ROOM_NAMES:
                if name in class_ious:
                    rows.append([name, f"{class_ious[name]:.4f}"])
            rows.append(["---", "---"])
            rows.append(["Mean IoU", f"{miou:.4f}"])
            rows.append(["Num Pairs", str(num_pairs)])

            print_table("MIoU Results (Color Matching)", rows, ["Room Type", "IoU"])
        except Exception as e:
            print(f"MIoU computation failed: {e}")

    # Summary
    if results:
        print("\n=== Summary ===")
        if "fid" in results:
            print(f"  FID:  {results['fid']:.4f}")
        if "miou" in results:
            print(f"  MIoU: {results['miou']:.4f}")
        print()


if __name__ == "__main__":
    main()
