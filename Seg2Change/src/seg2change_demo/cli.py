from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_base_image(size: int = 128) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (28, 46, 62))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((10, 14, 56, 60), fill=(76, 130, 88))
    draw.rectangle((72, 18, 112, 48), fill=(160, 168, 180))
    draw.rectangle((18, 76, 46, 112), fill=(194, 154, 96))
    draw.rectangle((62, 70, 118, 116), fill=(52, 88, 128))

    return canvas


def create_sample_triplet(output_dir: Path, size: int = 128) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    image_a = make_base_image(size)
    image_b = image_a.copy()
    label = Image.new("L", (size, size), 0)

    draw_b = ImageDraw.Draw(image_b)
    draw_label = ImageDraw.Draw(label)

    changed_box = (78, 22, 112, 56)
    draw_b.rectangle(changed_box, fill=(218, 92, 76))
    draw_label.rectangle(changed_box, fill=255)

    removed_box = (18, 76, 46, 112)
    draw_b.rectangle(removed_box, fill=(28, 46, 62))
    draw_label.rectangle(removed_box, fill=255)

    circle_bounds = (36, 30, 58, 52)
    draw_b.ellipse(circle_bounds, fill=(228, 226, 106))
    draw_label.ellipse(circle_bounds, fill=255)

    paths = {
        "A": output_dir / "A.png",
        "B": output_dir / "B.png",
        "label": output_dir / "label.png",
    }
    image_a.save(paths["A"])
    image_b.save(paths["B"])
    label.save(paths["label"])
    return paths


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return mask > 0


def infer_change_mask(image_a: np.ndarray, image_b: np.ndarray, threshold: int = 36) -> np.ndarray:
    diff = np.abs(image_b - image_a).mean(axis=2)
    return diff >= threshold


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "changed_pixels_pred": int(pred.sum()),
        "changed_pixels_gt": int(gt.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "iou": round(iou, 6),
        "f1": round(f1, 6),
    }


def write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def handle_generate_sample(args: argparse.Namespace) -> int:
    paths = create_sample_triplet(Path(args.output_dir), size=args.size)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


def handle_smoke_test(args: argparse.Namespace) -> int:
    output_dir = ensure_dir(Path(args.output_dir))

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = output_dir
        create_sample_triplet(input_dir, size=args.size)

    image_a = load_rgb(input_dir / "A.png")
    image_b = load_rgb(input_dir / "B.png")
    label = load_mask(input_dir / "label.png")

    pred = infer_change_mask(image_a, image_b, threshold=args.threshold)
    metrics = compute_metrics(pred, label)

    write_mask(output_dir / "pred_mask.png", pred)
    for name in ("A.png", "B.png", "label.png"):
        src = input_dir / name
        dst = output_dir / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


def validate_required_files(paths: list[Path]) -> list[str]:
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
    return missing


def handle_prepare_upstream_run(args: argparse.Namespace) -> int:
    upstream_root = Path(args.upstream_root)
    dataset_root = Path(args.dataset_root)
    weights_root = Path(args.weights_root)
    output_root = ensure_dir(Path(args.output_root))

    required = [
        upstream_root / "test_cach_ovcd.py",
        upstream_root / "train_cach_dino.py",
        weights_root / "sam3" / "sam3.pt",
        weights_root / "dinov2" / "dinov2_vitb14_pretrain.pth",
        weights_root / "cach" / "best.pth",
        dataset_root,
    ]
    missing = validate_required_files(required)
    if missing:
        print(json.dumps({"status": "missing_files", "missing": missing}, indent=2))
        return 1

    cmd = [
        "python",
        "test_cach_ovcd.py",
        "--checkpoint_path",
        str(weights_root / "cach" / "best.pth"),
        "--dataset_root_path",
        str(dataset_root) + "/",
        "--save_path",
        str(output_root),
        "--test_dataset",
        args.test_dataset,
    ]

    payload = {
        "status": "ready",
        "working_directory": str(upstream_root),
        "command": cmd,
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seg2Change helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-sample", help="Create synthetic sample inputs")
    sample_parser.add_argument("--output-dir", required=True)
    sample_parser.add_argument("--size", type=int, default=128)
    sample_parser.set_defaults(func=handle_generate_sample)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run the lightweight smoke test")
    smoke_parser.add_argument("--input-dir")
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.add_argument("--size", type=int, default=128)
    smoke_parser.add_argument("--threshold", type=int, default=36)
    smoke_parser.set_defaults(func=handle_smoke_test)

    prep_parser = subparsers.add_parser("prepare-upstream-run", help="Validate upstream full-run layout")
    prep_parser.add_argument("--upstream-root", required=True)
    prep_parser.add_argument("--dataset-root", required=True)
    prep_parser.add_argument("--weights-root", required=True)
    prep_parser.add_argument("--output-root", required=True)
    prep_parser.add_argument("--test-dataset", default="WHU-CD")
    prep_parser.set_defaults(func=handle_prepare_upstream_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
