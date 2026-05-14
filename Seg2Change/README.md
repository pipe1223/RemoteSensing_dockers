# Seg2Change

This folder packages a Docker-friendly Seg2Change workflow for the upstream project:

- Upstream repository: `https://github.com/yogurts-sy/Seg2Change`
- Paper: `Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing Change Detection`

## What is included

- `Dockerfile`: builds a lightweight Python image for the verified smoke-test workflow.
- `compose.yaml`: example service definition for local use.
- `requirements.txt`: minimal runtime requirements for the smoke test.
- `seg2change_demo.py`: runnable CLI bridge.
- `src/seg2change_demo/cli.py`: packaged CLI implementation.
- `tests/test_smoke.py`: local unit coverage for smoke-test, JSON-config, and annotation-input paths.
- `docs/implementation-report.md`: engineering notes, validation, and known limits.

## Why this package is structured this way

The upstream Seg2Change repository depends on a large vendored `sam3` codebase, CUDA-oriented PyTorch and MMCV installs, external checkpoints for SAM3 and DINOv2, and prepared datasets for evaluation. In this workspace, Docker itself was unavailable and the heavy ML stack was not preinstalled, so the practical deliverable is:

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

### 2a. Run annotation JSON input

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
- `--backend diff`: the current lightweight backend that computes a change mask from image differences

For each pair, the container writes:

- `A.png`
- `B.png`
- `pred_mask.png`
- `metrics.json`

And it also writes:

- `summary.json`

The current annotation reader supports:

- a top-level `pairs` list with `image_a` and `image_b`
- image records that directly contain paired path fields such as `image_a` and `image_b`
- grouped image records with keys like `pair_id` or `group_id`, when two images can be matched into a before/after pair

### 2b. Run annotation mode from a JSON config file

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

## Full upstream evaluation path

The real upstream evaluation flow still needs the original Seg2Change source tree plus checkpoints and datasets.

### Validate that layout

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  remote-sensing/seg2change:latest \
  prepare-upstream-run \
  --upstream-root /workspace/upstream/Seg2Change \
  --dataset-root /workspace/datasets/OVCD_Benchmark \
  --weights-root /workspace/upstream/Seg2Change/weights \
  --output-root /workspace/outputs \
  --test-dataset WHU-CD
```

## Local development without Docker

```bash
python seg2change_demo.py smoke-test --output-dir ./artifacts/smoke
python seg2change_demo.py --annotations ./annotations.json --image-root ./data --output-dir ./outputs --backend diff
python seg2change_demo.py --annotations ./annotations.json --image-root ./data --output-dir ./outputs --config-json ./config.json
python -m unittest discover -s tests
```

## Known limitations

- The annotation mode currently uses the lightweight `diff` backend rather than the full upstream Seg2Change model stack.
- Full Seg2Change evaluation was not executed in this workspace because Docker, CUDA, and the required checkpoints were unavailable.
- If your annotation JSON uses a completely custom pairing schema, the bridge may need one more parser rule for that dataset.
