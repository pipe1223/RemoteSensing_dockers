from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HARDCODED_CUDA_LINE = 'os.environ["CUDA_VISIBLE_DEVICES"] = "7"'


def sanitize_upstream_source(source: str) -> str:
    return source.replace(HARDCODED_CUDA_LINE, "# CUDA selection delegated to the wrapper")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the upstream Seg2Change evaluator without editing the upstream tree")
    parser.add_argument("--script", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("forwarded_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_path = Path(args.script).resolve()
    workdir = Path(args.workdir).resolve()
    forwarded_args = list(args.forwarded_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    source = sanitize_upstream_source(script_path.read_text(encoding="utf-8"))
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    sys.argv = [str(script_path), *forwarded_args]
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
    }
    exec(compile(source, str(script_path), "exec"), globals_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
