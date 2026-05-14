import json
from argparse import Namespace
import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "/workspace/RemoteSensing_dockers/Seg2Change/src")

from seg2change_demo.cli import (
    apply_json_overrides,
    build_pairs_from_annotations,
    compute_metrics,
    create_sample_triplet,
    infer_change_mask,
    load_mask,
    load_rgb,
    main,
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

            copy_targets = {
                "before.png": sample_dir / "A.png",
                "after.png": sample_dir / "B.png",
            }
            for target_name, source_path in copy_targets.items():
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


if __name__ == "__main__":
    unittest.main()
