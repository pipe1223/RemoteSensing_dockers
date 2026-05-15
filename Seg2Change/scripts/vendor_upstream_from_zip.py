from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

SKIP_EXTENSIONS = {
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".engine",
}

# These are not needed for running the Seg2Change Python source and make the repo heavy.
SKIP_DIR_PARTS = {
    ".git",
    "__pycache__",
}


def should_skip(member_name: str) -> bool:
    path = Path(member_name)
    if any(part in SKIP_DIR_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    return False


def strip_top_level(member_name: str) -> Path | None:
    parts = Path(member_name).parts
    if len(parts) <= 1:
        return None
    return Path(*parts[1:])


def vendor_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> dict[str, object]:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            relative = strip_top_level(info.filename)
            if relative is None:
                continue
            if should_skip(info.filename):
                skipped.append(info.filename)
                continue
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied.append(str(relative))

    manifest = {
        "source_zip": str(zip_path),
        "output_dir": str(output_dir),
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "skipped_extensions": sorted(SKIP_EXTENSIONS),
        "copied": copied,
        "skipped": skipped,
    }
    with (output_dir / "VENDORED_FROM_ZIP.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor upstream Seg2Change source from a ZIP while skipping model checkpoints.")
    parser.add_argument("zip_path", help="Path to Seg2Change-main.zip or an equivalent upstream ZIP archive")
    parser.add_argument(
        "--output-dir",
        default="upstream/Seg2Change",
        help="Where to write the unpacked upstream source relative to the current working directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing output directory before extracting")
    args = parser.parse_args()

    manifest = vendor_zip(Path(args.zip_path), Path(args.output_dir), overwrite=args.overwrite)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"copied", "skipped"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
