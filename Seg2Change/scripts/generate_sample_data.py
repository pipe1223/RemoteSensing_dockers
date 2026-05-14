from pathlib import Path

from seg2change_demo.cli import create_sample_triplet


def main() -> None:
    create_sample_triplet(Path("sample_data"))


if __name__ == "__main__":
    main()
