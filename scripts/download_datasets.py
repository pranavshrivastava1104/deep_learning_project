"""Download the public datasets required by Phase 2."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pipelines.data.downloader import DATASETS, DatasetError, download_datasets


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(*DATASETS, "all"),
        help="Dataset archive to download, or 'all' for every configured archive.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Destination root for downloaded datasets (default: data).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable download progress bars.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset downloader CLI."""

    args = _parse_args(argv)
    selected = cast(str, args.dataset)
    data_dir = cast(Path, args.data_dir)
    datasets = list(DATASETS) if selected == "all" else [selected]

    try:
        results = download_datasets(
            datasets,
            data_dir,
            show_progress=not cast(bool, args.no_progress),
        )
    except (DatasetError, OSError) as error:
        print(f"ERROR: {error}")
        return 1

    for result in results:
        status = "downloaded" if result.created else "already present"
        print(f"{result.dataset}: {status} at {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
