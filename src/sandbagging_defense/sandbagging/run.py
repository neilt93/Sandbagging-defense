"""Phase 3 entry point: run all sandbag conditions across MMLU/HumanEval/preferences.

Implementation lands in Phase 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 3 not yet implemented. Config: {args.config}")


if __name__ == "__main__":
    main()
