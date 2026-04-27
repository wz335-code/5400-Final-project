"""Download the public OpenSub-Slang benchmark metadata package."""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

from slang_normalizer.logging_utils import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "external_data" / "slang_llm_benchmark"
REPO_ZIP_URL = (
    "https://github.com/amazon-science/slang-llm-benchmark/archive/refs/heads/main.zip"
)
DATA_PREFIX = "slang-llm-benchmark-main/Data/"
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark metadata download."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the public slang benchmark repository archive and extract "
            "its Data/ folder."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the benchmark Data/ files should be extracted.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=REPO_ZIP_URL,
        help="Zip archive URL for the public benchmark repository.",
    )
    return parser.parse_args()


def download_zip_bytes(url: str) -> bytes:
    """Download the repository archive as raw bytes."""

    with urllib.request.urlopen(url) as response:
        return response.read()


def extract_data_folder(zip_bytes: bytes, output_dir: Path) -> list[Path]:
    """Extract the benchmark Data/ folder from the downloaded archive."""

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(DATA_PREFIX) or member.is_dir():
                continue

            relative_name = member.filename.removeprefix(DATA_PREFIX)
            target_path = output_dir / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member) as source_file, target_path.open("wb") as target:
                target.write(source_file.read())

            extracted_paths.append(target_path)

    return extracted_paths


def main() -> None:
    """Download and extract the public benchmark metadata package."""

    configure_logging()
    args = parse_args()
    logger.info("Downloading benchmark archive from %s", args.url)
    zip_bytes = download_zip_bytes(args.url)
    extracted_paths = extract_data_folder(zip_bytes, args.output_dir)
    logger.info("Extracted %d files to %s", len(extracted_paths), args.output_dir)
    print(f"Extracted {len(extracted_paths)} benchmark files to {args.output_dir}")


if __name__ == "__main__":
    main()
