# Seg2Change

This folder packages a Docker-friendly Seg2Change workflow for the upstream project:

- Upstream repository: `https://github.com/yogurts-sy/Seg2Change`
- Paper: `Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing Change Detection`

## What is in this package

- `Dockerfile`: builds a lightweight Python image for the verified smoke test workflow.
- `compose.yaml`: example service definition for local use.
- `src/seg2change_demo/cli.py`: runnable CLI for sample generation, smoke testing, and full-run path validation.
- `scripts/run_upstream_eval.py`: wrapper that launches the upstream evaluator without editing the upstream tree.
- `scripts/vendor_upstream_from_zip.py`: helper for unpacking the upstream Seg2Change ZIP into `upstream/Seg2Change` while skipping checkpoint binaries.
- `requirements.upstream.txt`: upstream Python dependency reference for the GPU-backed evaluation path.
- `tests/test_smoke.py`: local unit coverage for the smoke-test path.
- `docs/implementation-report.md`: engineering notes, decisions, validation, and limitations.

## Why this package is structured this way

The upstream Seg2Change repository is not a small standalone script. It depends on:

- a large vendored `sam3` codebase
- CUDA-oriented PyTorch and MMCV installs
- external checkpoints for SAM3, DINOv2, and the Seg2Change change head
- prepared datasets for evaluation and training

Inside this workspace, Docker itself was unavailable and the heavy ML stack was not preinstalled, so the practical deliverable is:

1. a verified runnable smoke test that exercises the container and I/O flow end to end
2. a lightweight annotation-driven entrypoint that can read JSON pair definitions
3. a documented, GPU-ready full-run path for the real upstream evaluation workflow

## Quick start

### 1. Build the image

```bash
docker build -t remote-sensing/seg2change:latest .
```

### 2. Run the smoke test

```bash
docker run --rm \
  -v "$(pwd)/artifacts:/workspace/artifacts" \
  remote-sensing/seg2change:latest \
  smoke-test \
  --output-dir /workspace/artifacts/smoke
```

This command will:

- generate a synthetic pair of bi-temporal sample images
- run a lightweight change-detection inference pass
- save a predicted mask and metrics JSON under `/workspace/artifacts/smoke`

### 2a. Run annotation JSON input

This is the container interface for dataset-backed runs:

```bash
docker run --rm \
  -v /absolute/path/to/dataset/root:/data:ro \
  -v /absolute/path/to/Annotations_test_change_detection.json:/annotations.json:ro \
  -v "$PWD/outputs":/outputs \
  remote-sensing/seg2change:latest \
  --annotations /annotations.json \
  --image-root /data \
  --output-dir /outputs \
  --backend diff
```

What it expects:

- `--annotations`: a JSON file that contains image pairs
- `--image-root`: the root folder used to resolve image paths from that JSON
- `--output-dir`: where masks and summaries should be written
- `--backend diff`: the lightweight backend that computes a change mask from image differences

For each pair, the container writes:

- `A.png`
- `B.png`
- `pred_mask.png`
- `metrics.json`

And it also writes:

- `summary.json`

The annotation reader supports:

- a top-level `pairs` list with `image_a` and `image_b`
- COCO-style `images` records where `file_name` is a two-item list, such as `[before_path, after_path]`
- image records that directly contain paired path fields such as `image_a` and `image_b`
- grouped image records with keys like `pair_id` or `group_id`, when two images can be matched into a before/after pair

### 2b. Run annotation mode with the real Seg2Change backend

`--backend seg2change` converts your JSON pairs into a temporary upstream-style dataset, then launches the upstream `test_cach_ovcd.py` evaluator through `scripts/run_upstream_eval.py`.

Before using this mode, make sure `upstream/Seg2Change` exists. You can create it from the uploaded upstream ZIP with:

```bash
python scripts/vendor_upstream_from_zip.py /path/to/Seg2Change-main.zip \
  --output-dir upstream/Seg2Change \
  --overwrite
```

The vendoring helper intentionally skips model/checkpoint binaries such as `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.onnx`, and `.engine`. Put required weights back under the expected upstream paths after extraction, for example:

```text
upstream/Seg2Change/weights/sam3/sam3.pt
upstream/Seg2Change/weights/dinov2/dinov2_vitb14_pretrain.pth
upstream/Seg2Change/weights/cach/best.pth
```

Dry-run first:

```bash
docker run --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PWD:/workspace" \
  -v /absolute/path/to/dataset/root:/data:ro \
  -v /absolute/path/to/Annotations_test_change_detection.json:/annotations.json:ro \
  remote-sensing/seg2change:latest \
  --annotations /annotations.json \
  --image-root /data \
  --output-dir /workspace/outputs \
  --backend seg2change \
  --upstream-root /workspace/upstream/Seg2Change \
  --test-dataset CLCD \
  --cuda-visible-devices 0 \
  --dry-run
```

Run the real GPU-backed evaluation by removing `--dry-run`:

```bash
docker run --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PWD:/workspace" \
  -v /absolute/path/to/dataset/root:/data:ro \
  -v /absolute/path/to/Annotations_test_change_detection.json:/annotations.json:ro \
  remote-sensing/seg2change:latest \
  --annotations /annotations.json \
  --image-root /data \
  --output-dir /workspace/outputs \
  --backend seg2change \
  --upstream-root /workspace/upstream/Seg2Change \
  --test-dataset CLCD \
  --cuda-visible-devices 0
```

### 2c. Run annotation mode from a JSON config file

Example `config.json`:

```json
{
  "annotations": "/workspace/annotations.json",
  "image_root": "/workspace/data",
  "output_dir": "/workspace/outputs",
  "backend": "diff",
  "threshold": 36
}
```

Run it like this:

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  remote-sensing/seg2change:latest \
  --annotations /workspace/annotations.json \
  --image-root /workspace/data \
  --output-dir /workspace/outputs \
  --config-json /workspace/config.json
```

### 3. Optional: use Compose

```bash
docker compose run --rm seg2change smoke-test --output-dir /workspace/artifacts/smoke
```

## Expected smoke-test outputs

The smoke test writes:

- `A.png`
- `B.png`
- `label.png`
- `pred_mask.png`
- `metrics.json`

Example metrics keys:

- `iou`
- `precision`
- `recall`
- `f1`
- `changed_pixels_pred`
- `changed_pixels_gt`

## Full upstream evaluation path

The real upstream evaluation flow still needs the original Seg2Change source tree plus checkpoints and datasets.
This package includes a wrapper that can execute the upstream `test_cach_ovcd.py` script directly from an unpacked upstream source folder.

What the wrapper does:

- runs the real upstream evaluator
- removes the upstream script's hardcoded `CUDA_VISIBLE_DEVICES="7"` line at runtime so you can choose the GPU from the container command
- writes `upstream-command.json` and, after execution, `upstream-run-result.json` under your chosen output folder

### Validate that layout

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  remote-sensing/seg2change:latest \
  prepare-upstream-run \
  --upstream-root /workspace/upstream/Seg2Change \
  --dataset-root /workspace/datasets/OVCD_Benchmark \
  --output-root /workspace/outputs \
  --test-dataset WHU-CD
```

The command checks for the expected files and writes the exact wrapped execution command to `/workspace/outputs/upstream-command.json`.

### Execute the real upstream evaluation

```bash
docker run --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$(pwd):/workspace" \
  remote-sensing/seg2change:latest \
  run-upstream-eval \
  --upstream-root /workspace/upstream/Seg2Change \
  --dataset-root /workspace/datasets/OVCD_Benchmark \
  --output-root /workspace/outputs \
  --test-dataset WHU-CD \
  --cuda-visible-devices 0
```

Notes:

- `--weights-root` is optional when your checkpoints live in `/workspace/upstream/Seg2Change/weights`
- `--encoder-size` supports `small` and `base`
- `--ovss-model` supports `SegEarth-OV3`, `SAM3`, and `SegEarth-OV1`
- `--dry-run` prints the wrapped command without launching the upstream evaluator

### Upstream dependency reference

The smoke-test image intentionally stays lightweight. For a full upstream run, use the upstream dependency set in `requirements.upstream.txt` together with the CUDA-linked PyTorch, torchvision, xformers, and MMCV versions described in the original upstream README.

## Local development without Docker

```bash
python seg2change_demo.py smoke-test --output-dir ./artifacts/smoke
python seg2change_demo.py --annotations ./annotations.json --image-root ./data --output-dir ./outputs --backend diff
python seg2change_demo.py --annotations ./annotations.json --image-root ./data --output-dir ./outputs --backend seg2change --upstream-root ./upstream/Seg2Change --test-dataset CLCD --dry-run
python seg2change_demo.py --annotations ./annotations.json --image-root ./data --output-dir ./outputs --config-json ./config.json
python seg2change_demo.py prepare-upstream-run --upstream-root ./upstream/Seg2Change --dataset-root ./datasets/OVCD_Benchmark --output-root ./outputs
python seg2change_demo.py run-upstream-eval --upstream-root ./upstream/Seg2Change --dataset-root ./datasets/OVCD_Benchmark --output-root ./outputs --test-dataset WHU-CD --dry-run
python -m unittest discover -s tests
```

## Known limitations

- The smoke test is a container and workflow verification harness, not a claim of reproduced paper metrics.
- Full Seg2Change evaluation was not executed in this workspace because CUDA and the required checkpoints were unavailable.
- The vendoring helper skips model/checkpoint binaries by design. Keep `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.onnx`, and `.engine` files outside git or restore them manually under the expected `weights/` paths.

## Files to read next

- `docs/implementation-report.md`
- `Dockerfile`
- `src/seg2change_demo/cli.py`
- `scripts/vendor_upstream_from_zip.py`
