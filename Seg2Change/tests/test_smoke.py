import json
from argparse import Namespace
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/workspace/RemoteSensing_dockers/Seg2Change/src")

from seg2change_demo.cli import (
    apply_json_overrides,
    build_pairs_from_annotations,
    build_upstream_run_payload,
    compute_metrics,
    create_sample_triplet,
    infer_change_mask,
    load_mask,
    load_rgb,
    main,
    prepare_seg2change_annotation_dataset,
)


class SmokeTestCase(unittest.TestCase):
    def test_build_pairs_from_coco_style_file_name_list(self) -> None:
        data = {
            "images": [
                {
                    "id": 7,
                    "file_name": [
                        "RGB/CDD/images/val+A+00540.jpg",
                        "RGB/CDD/change_images/val+B+00540.jpg",
                    ],
                    "width": 256,
                    "height": 256,
                }
            ],
            "annotations": [],
        }
        pairs = build_pairs_from_annotations(data)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["id"], 7)
        self.assertEqual(pairs[0]["image_a"], "RGB/CDD/images/val+A+00540.jpg")
        self.assertEqual(pairs[0]["image_b"], "RGB/CDD/change_images/val+B+00540.jpg")

    def test_annotations_mode_accepts_raw_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "images"
            input_dir.mkdir(parents=True, exist_ok=True)
            sample_dir = root / "pair"
            create_sample_triplet(sample_dir, size=128)

            shutil_targets = {
                "before.png": sample_dir / "A.png",
                "after.png": sample_dir / "B.png",
            }
            for target_name, source_path in shutil_targets.items():
                (input_dir / target_name).write_bytes(source_path.read_bytes())

            annotations_path = root / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "id": "sample-1",
                                "image_a": "before.png",
                                "image_b": "after.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            exit_code = main(
                [
                    "--annotations",
                    str(annotations_path),
                    "--image-root",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--backend",
                    "diff",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "sample-1" / "pred_mask.png").exists())
            self.assertTrue((output_dir / "summary.json").exists())

    def test_prepare_seg2change_annotation_dataset_creates_upstream_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "data"
            sample_dir = root / "sample"
            output_dir = root / "outputs"
            create_sample_triplet(sample_dir, size=128)
            (image_root / "RGB" / "CDD" / "images").mkdir(parents=True, exist_ok=True)
            (image_root / "RGB" / "CDD" / "change_images").mkdir(parents=True, exist_ok=True)
            image_a_rel = Path("RGB/CDD/images/test+A+00001.png")
            image_b_rel = Path("RGB/CDD/change_images/test+B+00001.png")
            (image_root / image_a_rel).write_bytes((sample_dir / "A.png").read_bytes())
            (image_root / image_b_rel).write_bytes((sample_dir / "B.png").read_bytes())
            data = {
                "images": [
                    {
                        "id": 1,
                        "file_name": [str(image_a_rel), str(image_b_rel)],
                        "width": 128,
                        "height": 128,
                    }
                ],
                "annotations": [
                    {
                        "image_id": 1,
                        "segmentation": [[10, 10, 64, 10, 64, 64, 10, 64]],
                    }
                ],
            }
            prepared = prepare_seg2change_annotation_dataset(
                pairs=build_pairs_from_annotations(data),
                annotations=data["annotations"],
                image_root=image_root,
                output_dir=output_dir,
                test_dataset="CLCD",
            )

            dataset_dir = Path(prepared["dataset_dir"])
            self.assertTrue((dataset_dir / "A" / "000000.png").exists())
            self.assertTrue((dataset_dir / "B" / "000000.png").exists())
            self.assertTrue((dataset_dir / "label" / "000000.png").exists())
            self.assertEqual(len(prepared["prepared_pairs"]), 1)

    def test_seg2change_annotation_backend_supports_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "images"
            input_dir.mkdir(parents=True, exist_ok=True)
            sample_dir = root / "pair"
            create_sample_triplet(sample_dir, size=128)
            (input_dir / "before.png").write_bytes((sample_dir / "A.png").read_bytes())
            (input_dir / "after.png").write_bytes((sample_dir / "B.png").read_bytes())

            annotations_path = root / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "id": "sample-1",
                                "image_a": "before.png",
                                "image_b": "after.png",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            upstream_root = root / "upstream" / "Seg2Change"
            dataset_root = root / "outputs" / "_seg2change_annotation_dataset"
            output_root = root / "outputs"
            required_dirs = [
                upstream_root / "dataset",
                upstream_root / "model" / "ovcd",
                upstream_root / "model" / "backbone",
                upstream_root / "configs",
                upstream_root / "weights" / "sam3",
                upstream_root / "weights" / "dinov2",
                upstream_root / "weights" / "cach",
            ]
            for path in required_dirs:
                path.mkdir(parents=True, exist_ok=True)
            required_files = [
                upstream_root / "test_cach_ovcd.py",
                upstream_root / "seg_model_sam3.py",
                upstream_root / "dataset" / "ovcd.py",
                upstream_root / "model" / "ovcd" / "change_head_fdr_dino.py",
                upstream_root / "model" / "backbone" / "dinov2.py",
                upstream_root / "weights" / "sam3" / "sam3.pt",
                upstream_root / "weights" / "dinov2" / "dinov2_vitb14_pretrain.pth",
                upstream_root / "weights" / "cach" / "best.pth",
            ]
            for path in required_files:
                path.write_text("stub", encoding="utf-8")

            exit_code = main(
                [
                    "--annotations",
                    str(annotations_path),
                    "--image-root",
                    str(input_dir),
                    "--output-dir",
                    str(output_root),
                    "--backend",
                    "seg2change",
                    "--upstream-root",
                    str(upstream_root),
                    "--test-dataset",
                    "CLCD",
                    "--dry-run",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "seg2change-run-command.json").exists())
            self.assertTrue((dataset_root / "CLCD-512" / "test" / "A" / "000000.png").exists())

    def test_metrics_are_reasonable_for_generated_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            create_sample_triplet(root, size=128)

            pred = infer_change_mask(load_rgb(root / "A.png"), load_rgb(root / "B.png"), threshold=36)
            gt = load_mask(root / "label.png")
            metrics = compute_metrics(pred, gt)

            self.assertGreater(metrics["iou"], 0.99)
            self.assertGreater(metrics["f1"], 0.99)

    def test_metrics_payload_is_json_serializable(self) -> None:
        payload = compute_metrics(pred=[[True, False]], gt=[[True, False]])
        json.dumps(payload)

    def test_cli_can_read_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            config_path = root / "smoke.json"
            create_sample_triplet(input_dir, size=128)
            config_path.write_text(
                json.dumps(
                    {
                        "input_dir": str(input_dir),
                        "output_dir": str(output_dir),
                        "threshold": 36,
                    }
                ),
                encoding="utf-8",
            )
            args = apply_json_overrides(
                Namespace(
                    config_json=str(config_path),
                    input_dir=None,
                    output_dir=str(output_dir),
                    threshold=10,
                    size=128,
                ),
                {"input_dir", "output_dir", "threshold", "size"},
            )
            self.assertEqual(args.input_dir, str(input_dir))
            self.assertEqual(args.output_dir, str(output_dir))
            self.assertEqual(args.threshold, 36)

    def test_upstream_payload_uses_wrapper_and_default_weight_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            upstream_root = root / "upstream" / "Seg2Change"
            dataset_root = root / "datasets" / "OVCD_Benchmark"
            output_root = root / "outputs"

            required_dirs = [
                upstream_root / "dataset",
                upstream_root / "model" / "ovcd",
                upstream_root / "model" / "backbone",
                upstream_root / "configs",
                upstream_root / "weights" / "sam3",
                upstream_root / "weights" / "dinov2",
                upstream_root / "weights" / "cach",
                dataset_root,
            ]
            for path in required_dirs:
                path.mkdir(parents=True, exist_ok=True)

            required_files = [
                upstream_root / "test_cach_ovcd.py",
                upstream_root / "seg_model_sam3.py",
                upstream_root / "dataset" / "ovcd.py",
                upstream_root / "model" / "ovcd" / "change_head_fdr_dino.py",
                upstream_root / "model" / "backbone" / "dinov2.py",
                upstream_root / "weights" / "sam3" / "sam3.pt",
                upstream_root / "weights" / "dinov2" / "dinov2_vitb14_pretrain.pth",
                upstream_root / "weights" / "cach" / "best.pth",
            ]
            for path in required_files:
                path.write_text("stub", encoding="utf-8")

            payload = build_upstream_run_payload(
                Namespace(
                    upstream_root=str(upstream_root),
                    dataset_root=str(dataset_root),
                    output_root=str(output_root),
                    weights_root=None,
                    checkpoint_path=None,
                    feat_root=None,
                    test_dataset="WHU-CD",
                    batch_size=1,
                    crop_size=504,
                    encoder_size="base",
                    dino_ft="frozen",
                    ovss_model="SegEarth-OV3",
                    cuda_visible_devices="0",
                )
            )

            self.assertEqual(payload["status"], "ready")
            self.assertIn("run_upstream_eval.py", payload["shell_command"])
            self.assertIn("--cuda-visible-devices 0", payload["shell_command"])
            self.assertIn("--test_dataset WHU-CD", payload["shell_command"])
            self.assertTrue(str(output_root / "features") in payload["shell_command"])


if __name__ == "__main__":
    unittest.main()
