# Seg2Change

This folder packages a Docker-friendly Seg2Change workflow for the upstream project:

- Upstream repository: `https://github.com/yogurts-sy/Seg2Change`
- Paper: `Seg2Change: Adapting Open-Vocabulary Semantic Segmentation Model for Remote Sensing Change Detection`

## What is included

- `Dockerfile`: builds a lightweight Python image for the verified smoke-test workflow.
- `compose.yaml`: example service definition for local use.
- `requirements.txt`: minimal runtime requirements for the smoke test.
- `seg2change_demo.py`: runnable CLI for sample generation, smoke testing, and full-run path validation.
- `implementation_report.md`: engineering notes, validation, and known limits.

## Why this package is structured this way

The upstream Seg2Change repository depends on a large vendored `sam3` codebase, CUDA-oriented PyTorch and MMCV installs, external checkpoints for SAM3 and DINOv2, and prepared datasets for evaluation. In this workspace, Docker itself was unavailable and the heavy ML stack was not preinstalled, so the practical deliverable is:

1. a verified runnable smoke test that exercises the container and I/O flow end to end
2. a documented, GPU-ready full-run path for the real upstream evaluation workflow

## Quick start

### Build the image

```bash
docker build -t seg2change-demo .
```

### Run the smoke test

```bash
docker run --rm \
  -v "$(pwd)/artifacts:/workspace/artifacts" \
  seg2change-demo \
  smoke-test \
  --output-dir /workspace/artifacts/smoke
```

The command generates a synthetic bi-temporal image pair, runs a lightweight change-detection inference pass, and writes these files under `/workspace/artifacts/smoke`:

- `A.png`
- `B.png`
- `label.png`
- `pred_mask.png`
- `metrics.json`

### Optional: use Compose

```bash
docker compose run --rm seg2change smoke-test --output-dir /workspace/artifacts/smoke
```

## Full upstream evaluation path

The real upstream evaluation flow still needs the original Seg2Change source tree plus checkpoints and datasets.

### Expected mounted layout

```text
Seg2Change/
  upstream/
    Seg2Change/
      test_cach_ovcd.py
      train_cach_dino.py
      weights/
        sam3/
          sam3.pt
        dinov2/
          dinov2_vitb14_pretrain.pth
        cach/
          best.pth
  datasets/
    OVCD_Benchmark/
  outputs/
```

### Validate that layout

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  seg2change-demo \
  prepare-upstream-run \
  --upstream-root /workspace/upstream/Seg2Change \
  --dataset-root /workspace/datasets/OVCD_Benchmark \
  --weights-root /workspace/upstream/Seg2Change/weights \
  --output-root /workspace/outputs \
  --test-dataset WHU-CD
```

The command checks for the expected files and prints the exact upstream evaluation command to run next.

## Known limitations

- The smoke test is a container and workflow verification harness, not a reproduced paper benchmark.
- Full Seg2Change evaluation was not executed in this workspace because Docker, CUDA, and the required checkpoints were unavailable.
- The upstream repository uses additional vendored modules beyond what is needed for the smoke test here.
