# Seg2Change Implementation Report

## Goal

Create a Docker-oriented Seg2Change package inside `RemoteSensing_dockers` with:

- a dedicated `Seg2Change` folder
- a model-specific README
- a root README for the repository
- runnable code that can be verified in the current workspace

## Source assessment

The upstream repository is a research codebase centered on:

- `test_cach_ovcd.py` for evaluation
- `train_cach_dino.py` for training
- a DINOv2-based change head under `model/ovcd`
- a large vendored `sam3` implementation used by `seg_model_sam3.py`

The upstream project also requires:

- CUDA-oriented PyTorch
- MMCV and MMSegmentation compatibility
- external checkpoints for SAM3, DINOv2, and the trained change head
- prepared remote-sensing datasets

## Deliverable decision

A full reproduction could not be verified here because:

- normal GitHub cloning from the workspace was blocked
- Docker was not installed in the workspace
- the host Python environment did not include PyTorch
- the required model checkpoints and datasets were not available

Because of that, the implemented result is the smallest faithful runnable package:

1. a Docker-ready Seg2Change folder
2. a verified smoke-test CLI that exercises a change-detection workflow end to end
3. a documented and executable wrapper path for the real upstream evaluation workflow when the upstream checkout, checkpoints, and GPU runtime are mounted into the container

## Additional update

The uploaded upstream `Seg2Change` repository snapshot is now vendored in this package as `vendor/Seg2Change-main.zip`.

During Docker build, that archive is unpacked into `/app/upstream/Seg2Change`, and the uploaded `exp/CK/best.pth` is mirrored into `weights/cach/best.pth` for the wrapped evaluator.

This changes the practical run model in two ways:

- the helper image no longer requires a separate mounted upstream source checkout for the real Seg2Change path
- annotation-driven runs can now use `--backend seg2change` to prepare a temporary upstream-style dataset and launch the real upstream evaluator through the wrapper

## Files added

- `README.md`
- `Seg2Change/README.md`
- `Seg2Change/Dockerfile`
- `Seg2Change/compose.yaml`
- `Seg2Change/.dockerignore`
- `Seg2Change/requirements.txt`
- `Seg2Change/src/seg2change_demo/__init__.py`
- `Seg2Change/src/seg2change_demo/cli.py`
- `Seg2Change/tests/test_smoke.py`
- `Seg2Change/scripts/run_upstream_eval.py`
- `Seg2Change/docs/implementation-report.md`
- `Seg2Change/vendor/Seg2Change-main.zip`

## Validation performed

The following checks were run in the workspace:

- local Python smoke test using the new CLI
- local unit test execution with `unittest`
- local dry-run validation of the new upstream command builder
- local validation that annotation JSON can be prepared into an upstream-style dataset layout for the real Seg2Change flow

The following checks were not possible in the workspace:

- `docker build`
- `docker run`
- full upstream Seg2Change evaluation with checkpoints, datasets, CUDA, and the mounted GPU runtime

## Practical outcome

The delivered package is immediately usable for:

- repository organization
- onboarding
- container setup review
- smoke-test verification
- annotation-backed real Seg2Change runs once DINOv2 and SAM3 weights are mounted into the container
