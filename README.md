# RemoteSensing Dockers

Dockerized remote-sensing model runners and bridge code.

## Models

| Model | Folder | Status | Input | Output |
| --- | --- | --- | --- | --- |
| Seg2Change | [`Seg2Change/`](Seg2Change/) | Available | JSON annotation file where each `images[*].file_name` contains two image paths | Binary change masks, overlays, manifest files |

## Repository layout

```text
RemoteSensing_dockers/
├── README.md
└── Seg2Change/
    ├── Dockerfile
    ├── README.md
    ├── compose.yaml
    ├── requirements.txt
    ├── seg2change_json_infer.py
    ├── examples/
    └── tests/
```

## Quick start for Seg2Change

```bash
cd Seg2Change

docker build -t remote-sensing/seg2change:latest .

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

See [`Seg2Change/README.md`](Seg2Change/README.md) for the JSON format, validation mode, production Seg2Change backend notes, and output structure.
