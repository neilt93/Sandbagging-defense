"""Regenerate all paper figures from results/.

Implementation lands incrementally; the money plot is the Phase 3 output.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Figures not yet implemented. Out: {args.out}")


if __name__ == "__main__":
    main()
