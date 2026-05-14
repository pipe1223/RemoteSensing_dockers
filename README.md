# RemoteSensing_dockers

This repository hosts Docker-oriented packaging for remote sensing models.

## Available models

| Model | Status | What is included |
| --- | --- | --- |
| Seg2Change | Ready | A Docker-ready wrapper, a verified smoke-test workflow, and a wrapped upstream evaluation path for user-supplied checkpoints, datasets, and a mounted upstream checkout. |

## Repository layout

- `Seg2Change/`: container files, runnable demo code, tests, and usage documentation for Seg2Change.

## Notes

The current repository contains one model package: `Seg2Change`.
The Seg2Change package is designed to be useful in two ways:

1. It provides a lightweight smoke test that runs end to end and verifies the container wiring.
2. It can validate and launch the full upstream Seg2Change evaluation workflow when the original source tree, checkpoints, datasets, and GPU runtime are available.
