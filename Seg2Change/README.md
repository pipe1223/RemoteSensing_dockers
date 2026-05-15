# Seg2Change

This folder packages a Docker-friendly Seg2Change workflow for the upstream project:

- Upstream repository: `https://github.com/yogurts-sy/Seg2Change`
- Paper: `Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing Change Detection`

## What is in this package

- `Dockerfile`: builds the helper image and unpacks the vendored upstream Seg2Change source snapshot into `/app/upstream/Seg2Change`.
- `vendor/Seg2Change-main.zip`: vendored upstream repository snapshot from the uploaded archive.
- `src/seg2change_demo/cli.py`: runnable CLI for sample generation, smoke testing, annotation input, and full-run path validation.
- `scripts/run_upstream_eval.py`: wrapper that launches the upstream evaluator without editing the upstream tree.
- `tests/test_smoke.py`: local unit coverage for the smoke-test path and the annotation-backed upstream bridge.
- `docs/implementation-report.md`: engineering notes, decisions, validation, and limitations.

## Why this package is structured this way

The upstream Seg2Change repository is not a small standalone script. It depends on:

- a large vendored `sam3` codebase
- CUDA-oriented PyTorch and MMCV installs
- external checkpoints for SAM3, DINOv2, and the Seg2Change change head
- prepared datasets for evaluation and training

Inside this package, the practical deliverable is:

1. a verified runnable smoke test that exercises the container and I/O flow end to end
2. a lightweight annotation-driven entrypoint that can read JSON pair definitions
3. a real `--backend seg2change` path that converts paired-image annotation JSON into an upstream-style dataset layout and launches the vendored Seg2Change evaluator

## Vendored upstream layout

The uploaded upstream Seg2Change repository is now stored in the repo as:

```text
Seg2Change/vendor/Seg2Change-main.zip
```

During `docker build`, that archive is unpacked into this in-image path:

```text
/app/upstream/Seg2Change
```

What is already included there:

- upstream source code
- the uploaded `exp/CK/best.pth`
- `weights/cach/best.pth` copied from that uploaded checkpoint for the wrapped evaluator

What you still need to provide before a real GPU run:

- `weights/dinov2/dinov2_vitb14_pretrain.pth`
- `weights/sam3/sam3.pt`

Placeholder README files are included in those weight folders so the expected locations are visible in the repo.

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

### 3. Run annotation JSON with the lightweight backend

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

### 4. Run annotation JSON with the real Seg2Change backend

This path uses the vendored upstream code inside the image and prepares a temporary upstream-style dataset under your output directory.

Dry run first:

```bash
docker run --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /absolute/path/to/dataset/root:/data:ro \
  -v /absolute/path/to/Annotations_test_change_detection.json:/annotations.json:ro \
  -v /absolute/path/to/dinov2_vitb14_pretrain.pth:/app/upstream/Seg2Change/weights/dinov2/dinov2_vitb14_pretrain.pth:ro \
  -v /absolute/path/to/sam3.pt:/app/upstream/Seg2Change/weights/sam3/sam3.pt:ro \
  -v "$PWD/outputs":/outputs \
  remote-sensing/seg2change:latest \
  --annotations /annotations.json \
  --image-root /data \
  --output-dir /outputs \
  --backend seg2change \
  --cuda-visible-devices 0 \
  --test-dataset CLCD \
  --dry-run
```

Then the real run:

```bash
docker run --rm \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /absolute/path/to/dataset/root:/data:ro \
  -v /absolute/path/to/Annotations_test_change_detection.json:/annotations.json:ro \
  -v /absolute/path/to/dinov2_vitb14_pretrain.pth:/app/upstream/Seg2Change/weights/dinov2/dinov2_vitb14_pretrain.pth:ro \
  -v /absolute/path/to/sam3.pt:/app/upstream/Seg2Change/weights/sam3/sam3.pt:ro \
  -v "$PWD/outputs":/outputs \
  remote-sensing/seg2change:latest \
  --annotations /annotations.json \
  --image-root /data \
  --output-dir /outputs \
  --backend seg2change \
  --cuda-visible-devices 0 \
  --test-dataset CLCD
```
