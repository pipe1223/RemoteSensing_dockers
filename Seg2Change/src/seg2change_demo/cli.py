from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_config(config_json: str | None) -> dict[str, object]:
    if not config_json:
        return {}
    with Path(config_json).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object")
    return data


def apply_json_overrides(args: argparse.Namespace, allowed_keys: set[str]) -> argparse.Namespace:
    config = load_json_config(getattr(args, "config_json", None))
    for key in allowed_keys:
        if key in config:
            setattr(args, key, config[key])
    return args


def normalize_cli_args(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    known_commands = {
        "generate-sample",
        "smoke-test",
        "prepare-upstream-run",
        "run-upstream-eval",
        "run-annotations",
    }
    if argv[0] in known_commands:
        return argv
    if "--annotations" in argv:
        return ["run-annotations", *argv]
    return argv


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
    args = apply_json_overrides(args, {"input_dir", "output_dir", "size", "threshold"})
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


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else value + "/"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def resolve_under_root(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def find_first(record: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def find_pair_list(record: dict[str, object], keys: tuple[str, ...]) -> tuple[str, str] | None:
    for key in keys:
        value = record.get(key)
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], str)
            and isinstance(value[1], str)
        ):
            return value[0], value[1]
    return None


def build_pairs_from_annotations(data: dict[str, object]) -> list[dict[str, object]]:
    pair_records = data.get("pairs")
    if isinstance(pair_records, list):
        return [record for record in pair_records if isinstance(record, dict)]

    images = data.get("images")
    if not isinstance(images, list):
        return []

    image_records = [record for record in images if isinstance(record, dict)]
    pairs: list[dict[str, object]] = []

    before_keys = (
        "image_a",
        "imageA",
        "before_image",
        "before",
        "t1",
        "img1",
        "source",
        "reference",
        "file_name_a",
        "file_name_t1",
        "pre_image",
        "pre",
    )
    after_keys = (
        "image_b",
        "imageB",
        "after_image",
        "after",
        "t2",
        "img2",
        "target",
        "query",
        "file_name_b",
        "file_name_t2",
        "post_image",
        "post",
        "file_name",
    )
    pair_list_keys = ("file_name", "files", "image_paths", "pair")

    for record in image_records:
        pair_list = find_pair_list(record, pair_list_keys)
        if pair_list is not None:
            image_a, image_b = pair_list
            pairs.append(
                {
                    "id": record.get("id", record.get("pair_id", len(pairs))),
                    "image_a": image_a,
                    "image_b": image_b,
                    "record": record,
                }
            )
            continue

        image_a = find_first(record, before_keys)
        image_b = find_first(record, after_keys)
        if image_a and image_b:
            pairs.append(
                {
                    "id": record.get("id", record.get("pair_id", len(pairs))),
                    "image_a": image_a,
                    "image_b": image_b,
                    "record": record,
                }
            )

    if pairs:
        return pairs

    grouped: dict[str, list[dict[str, object]]] = {}
    for record in image_records:
        group_key = None
        for key in ("pair_id", "group_id", "group", "scene_id", "sample_id"):
            value = record.get(key)
            if value is not None:
                group_key = str(value)
                break
        if group_key is None:
            continue
        grouped.setdefault(group_key, []).append(record)

    phase_rank = {
        "before": 0,
        "pre": 0,
        "t1": 0,
        "a": 0,
        "img1": 0,
        "after": 1,
        "post": 1,
        "t2": 1,
        "b": 1,
        "img2": 1,
    }

    for group_key, records in grouped.items():
        if len(records) < 2:
            continue

        def record_rank(record: dict[str, object]) -> tuple[int, str]:
            phase = str(record.get("phase", record.get("time", record.get("split_role", "")))).lower()
            rank = phase_rank.get(phase, 99)
            file_name = str(record.get("file_name", ""))
            return rank, file_name

        ordered = sorted(records, key=record_rank)
        first = ordered[0]
        second = ordered[1]
        file_a = find_first(first, ("file_name",))
        file_b = find_first(second, ("file_name",))
        if file_a and file_b:
            pairs.append({"id": group_key, "image_a": file_a, "image_b": file_b, "record": {"group": group_key}})

    return pairs


def summarize_annotation_schema(data: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {"top_level_keys": sorted(data.keys())}
    images = data.get("images")
    if isinstance(images, list):
        summary["images_count"] = len(images)
        first_image = next((record for record in images if isinstance(record, dict)), None)
        if first_image is not None:
            summary["first_image_keys"] = sorted(first_image.keys())
            summary["first_image_sample"] = {
                key: first_image[key]
                for key in sorted(first_image.keys())[:8]
            }
    annotations = data.get("annotations")
    if isinstance(annotations, list):
        summary["annotations_count"] = len(annotations)
        first_annotation = next((record for record in annotations if isinstance(record, dict)), None)
        if first_annotation is not None:
            summary["first_annotation_keys"] = sorted(first_annotation.keys())
    pairs = data.get("pairs")
    if isinstance(pairs, list):
        summary["pairs_count"] = len(pairs)
        first_pair = next((record for record in pairs if isinstance(record, dict)), None)
        if first_pair is not None:
            summary["first_pair_keys"] = sorted(first_pair.keys())
    return summary


def save_annotation_outputs(
    output_dir: Path,
    pair_id: str,
    image_a_path: Path,
    image_b_path: Path,
    pred_mask: np.ndarray,
    metrics: dict[str, float | int],
    gt_mask: np.ndarray | None = None,
) -> None:
    pair_dir = ensure_dir(output_dir / str(pair_id))
    shutil.copy2(image_a_path, pair_dir / "A.png")
    shutil.copy2(image_b_path, pair_dir / "B.png")
    write_mask(pair_dir / "pred_mask.png", pred_mask)
    if gt_mask is not None:
        write_mask(pair_dir / "label.png", gt_mask)
    with (pair_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def build_gt_mask_from_annotations(
    annotations: list[dict[str, object]],
    image_id: int | str,
    width: int,
    height: int,
) -> np.ndarray | None:
    relevant = [ann for ann in annotations if ann.get("image_id") == image_id]
    if not relevant:
        return None

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for ann in relevant:
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            polygons = segmentation
            if polygons and all(isinstance(x, (int, float)) for x in polygons):
                polygons = [polygons]
            for polygon in polygons:
                if (
                    isinstance(polygon, list)
                    and len(polygon) >= 6
                    and all(isinstance(x, (int, float)) for x in polygon)
                ):
                    xy = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)]
                    draw.polygon(xy, fill=255)

    return np.asarray(canvas, dtype=np.uint8) > 0


def get_upstream_weights_root(args: argparse.Namespace, upstream_root: Path) -> Path:
    configured = getattr(args, "weights_root", None)
    return Path(configured) if configured else upstream_root / "weights"


def get_dino_checkpoint_name(encoder_size: str) -> str:
    mapping = {
        "small": "dinov2_vits14_pretrain.pth",
        "base": "dinov2_vitb14_pretrain.pth",
    }
    if encoder_size not in mapping:
        raise ValueError(f"Unsupported encoder size: {encoder_size}")
    return mapping[encoder_size]


def build_upstream_run_payload(args: argparse.Namespace) -> dict[str, object]:
    upstream_root = Path(args.upstream_root)
    dataset_root = Path(args.dataset_root)
    output_root = ensure_dir(Path(args.output_root))
    weights_root = get_upstream_weights_root(args, upstream_root)
    feat_root = Path(getattr(args, "feat_root", "") or (output_root / "features"))
    checkpoint_path = Path(getattr(args, "checkpoint_path", "") or (weights_root / "cach" / "best.pth"))
    wrapper_script = Path(__file__).resolve().parents[2] / "scripts" / "run_upstream_eval.py"
    test_script = upstream_root / "test_cach_ovcd.py"

    required = [
        wrapper_script,
        test_script,
        upstream_root / "dataset" / "ovcd.py",
        upstream_root / "model" / "ovcd" / "change_head_fdr_dino.py",
        upstream_root / "model" / "backbone" / "dinov2.py",
        upstream_root / "configs",
        dataset_root,
        checkpoint_path,
        weights_root / "dinov2" / get_dino_checkpoint_name(args.encoder_size),
    ]

    if args.ovss_model in {"SegEarth-OV3", "SAM3"}:
        required.extend(
            [
                upstream_root / "seg_model_sam3.py",
                weights_root / "sam3" / "sam3.pt",
            ]
        )
    elif args.ovss_model == "SegEarth-OV1":
        required.append(upstream_root / "segearthov1_segmentor.py")
    else:
        raise ValueError(f"Unsupported ovss model: {args.ovss_model}")

    missing = validate_required_files(required)

    runner_args = [
        "--checkpoint_path",
        str(checkpoint_path),
        "--dataset_root_path",
        ensure_trailing_slash(str(dataset_root)),
        "--save_path",
        str(output_root),
        "--test_dataset",
        args.test_dataset,
        "--encoder_size",
        args.encoder_size,
        "--dino_ft",
        args.dino_ft,
        "--crop_size",
        str(args.crop_size),
        "--batch_size",
        str(args.batch_size),
        "--ovss_model",
        args.ovss_model,
        "--feat_path",
        str(feat_root),
    ]
    command = [
        sys.executable,
        str(wrapper_script),
        "--script",
        str(test_script),
        "--workdir",
        str(upstream_root),
    ]
    if getattr(args, "cuda_visible_devices", None):
        command.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
    command.extend(["--", *runner_args])

    payload: dict[str, object] = {
        "status": "ready" if not missing else "missing_files",
        "upstream_root": str(upstream_root),
        "dataset_root": str(dataset_root),
        "weights_root": str(weights_root),
        "checkpoint_path": str(checkpoint_path),
        "feature_root": str(feat_root),
        "output_root": str(output_root),
        "working_directory": str(upstream_root),
        "script": str(test_script),
        "command": command,
        "shell_command": shell_join(command),
        "cuda_visible_devices": getattr(args, "cuda_visible_devices", None),
    }
    if missing:
        payload["missing"] = missing
    return payload


def handle_run_annotations(args: argparse.Namespace) -> int:
    args = apply_json_overrides(
        args,
        {"annotations", "image_root", "output_dir", "backend", "threshold"},
    )
    if args.backend != "diff":
        raise ValueError(
            "Unsupported backend for annotation mode: "
            f"{args.backend}. Use backend=diff for paired-image JSON input, "
            "or use run-upstream-eval for the real upstream Seg2Change evaluation flow."
        )

    annotations_path = Path(args.annotations)
    image_root = Path(args.image_root)
    output_dir = ensure_dir(Path(args.output_dir))

    missing = validate_required_files([annotations_path, image_root])
    if missing:
        print(json.dumps({"status": "missing_files", "missing": missing}, indent=2))
        return 1

    with annotations_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Annotations JSON must be an object")

    pairs = build_pairs_from_annotations(data)
    if not pairs:
        schema_summary = summarize_annotation_schema(data)
        raise ValueError(
            "Could not find image pairs in the annotations JSON. "
            f"Schema summary: {json.dumps(schema_summary, ensure_ascii=True)}"
        )

    raw_annotations = data.get("annotations")
    annotations = [ann for ann in raw_annotations if isinstance(ann, dict)] if isinstance(raw_annotations, list) else []

    summary: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        pair_id = str(pair.get("id", index))
        image_a_path = resolve_under_root(image_root, str(pair["image_a"]))
        image_b_path = resolve_under_root(image_root, str(pair["image_b"]))

        pair_missing = validate_required_files([image_a_path, image_b_path])
        if pair_missing:
            summary.append({"id": pair_id, "status": "missing_images", "missing": pair_missing})
            continue

        image_a = load_rgb(image_a_path)
        image_b = load_rgb(image_b_path)
        pred_mask = infer_change_mask(image_a, image_b, threshold=args.threshold)
        metrics: dict[str, float | int] = {
            "changed_pixels_pred": int(pred_mask.sum()),
            "threshold": args.threshold,
            "backend": args.backend,
        }
        record = pair.get("record", {})
        image_id = record.get("id") if isinstance(record, dict) else None
        gt_mask = None
        if image_id is not None:
            gt_mask = build_gt_mask_from_annotations(annotations, image_id, image_a.shape[1], image_a.shape[0])
            if gt_mask is not None:
                metrics.update(compute_metrics(pred_mask, gt_mask))
        save_annotation_outputs(output_dir, pair_id, image_a_path, image_b_path, pred_mask, metrics, gt_mask=gt_mask)
        summary.append({"id": pair_id, "status": "ok", "output_dir": str(output_dir / pair_id), **metrics})

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({"processed": len(summary), "summary_path": str(output_dir / "summary.json")}, indent=2))
    return 0


def handle_prepare_upstream_run(args: argparse.Namespace) -> int:
    args = apply_json_overrides(
        args,
        {
            "upstream_root",
            "dataset_root",
            "weights_root",
            "output_root",
            "test_dataset",
            "checkpoint_path",
            "feat_root",
            "batch_size",
            "crop_size",
            "encoder_size",
            "dino_ft",
            "ovss_model",
            "cuda_visible_devices",
        },
    )
    payload = build_upstream_run_payload(args)
    write_json(Path(args.output_root) / "upstream-command.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "ready" else 1


def handle_run_upstream_eval(args: argparse.Namespace) -> int:
    args = apply_json_overrides(
        args,
        {
            "upstream_root",
            "dataset_root",
            "weights_root",
            "output_root",
            "test_dataset",
            "checkpoint_path",
            "feat_root",
            "batch_size",
            "crop_size",
            "encoder_size",
            "dino_ft",
            "ovss_model",
            "cuda_visible_devices",
            "dry_run",
        },
    )
    payload = build_upstream_run_payload(args)
    metadata_path = Path(args.output_root) / "upstream-command.json"
    write_json(metadata_path, payload)
    if payload["status"] != "ready" or args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0 if args.dry_run and payload["status"] == "ready" else int(payload["status"] != "ready")

    completed = subprocess.run(payload["command"], cwd=str(payload["working_directory"]), check=False)
    result = {
        **payload,
        "returncode": completed.returncode,
        "metadata_path": str(metadata_path),
    }
    write_json(Path(args.output_root) / "upstream-run-result.json", result)
    print(json.dumps(result, indent=2))
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seg2Change helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-sample", help="Create synthetic sample inputs")
    sample_parser.add_argument("--output-dir", required=True)
    sample_parser.add_argument("--size", type=int, default=128)
    sample_parser.set_defaults(func=handle_generate_sample)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run the lightweight smoke test")
    smoke_parser.add_argument("--config-json")
    smoke_parser.add_argument("--input-dir")
    smoke_parser.add_argument("--output-dir", required=True)
    smoke_parser.add_argument("--size", type=int, default=128)
    smoke_parser.add_argument("--threshold", type=int, default=36)
    smoke_parser.set_defaults(func=handle_smoke_test)

    prep_parser = subparsers.add_parser("prepare-upstream-run", help="Validate upstream full-run layout")
    prep_parser.add_argument("--config-json")
    prep_parser.add_argument("--upstream-root", required=True)
    prep_parser.add_argument("--dataset-root", required=True)
    prep_parser.add_argument("--weights-root")
    prep_parser.add_argument("--output-root", required=True)
    prep_parser.add_argument("--test-dataset", default="WHU-CD")
    prep_parser.add_argument("--checkpoint-path")
    prep_parser.add_argument("--feat-root")
    prep_parser.add_argument("--batch-size", type=int, default=1)
    prep_parser.add_argument("--crop-size", type=int, default=504)
    prep_parser.add_argument("--encoder-size", default="base")
    prep_parser.add_argument("--dino-ft", default="frozen")
    prep_parser.add_argument("--ovss-model", default="SegEarth-OV3")
    prep_parser.add_argument("--cuda-visible-devices")
    prep_parser.set_defaults(func=handle_prepare_upstream_run)

    upstream_parser = subparsers.add_parser("run-upstream-eval", help="Execute the mounted upstream Seg2Change evaluator")
    upstream_parser.add_argument("--config-json")
    upstream_parser.add_argument("--upstream-root", required=True)
    upstream_parser.add_argument("--dataset-root", required=True)
    upstream_parser.add_argument("--weights-root")
    upstream_parser.add_argument("--output-root", required=True)
    upstream_parser.add_argument("--test-dataset", default="WHU-CD")
    upstream_parser.add_argument("--checkpoint-path")
    upstream_parser.add_argument("--feat-root")
    upstream_parser.add_argument("--batch-size", type=int, default=1)
    upstream_parser.add_argument("--crop-size", type=int, default=504)
    upstream_parser.add_argument("--encoder-size", default="base")
    upstream_parser.add_argument("--dino-ft", default="frozen")
    upstream_parser.add_argument("--ovss-model", default="SegEarth-OV3")
    upstream_parser.add_argument("--cuda-visible-devices")
    upstream_parser.add_argument("--dry-run", action="store_true")
    upstream_parser.set_defaults(func=handle_run_upstream_eval)

    ann_parser = subparsers.add_parser("run-annotations", help="Run annotation-driven change detection")
    ann_parser.add_argument("--config-json")
    ann_parser.add_argument("--annotations", required=True)
    ann_parser.add_argument("--image-root", required=True)
    ann_parser.add_argument("--output-dir", required=True)
    ann_parser.add_argument("--backend", default="diff")
    ann_parser.add_argument("--threshold", type=int, default=36)
    ann_parser.set_defaults(func=handle_run_annotations)

    return parser


def main(argv: list[str] | None = None) -> int:
    normalized_argv = normalize_cli_args(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(normalized_argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
