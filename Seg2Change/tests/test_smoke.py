import json
import tempfile
import unittest
from pathlib import Path

from seg2change_demo.cli import compute_metrics, create_sample_triplet, infer_change_mask, load_mask, load_rgb


class SmokeTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
