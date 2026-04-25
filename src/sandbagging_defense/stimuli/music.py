"""Phase 5: Bach chorale loader + stripping to pitch-class integer sequences.

Implementation lands in Phase 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 5 not yet implemented. Out: {args.out}")


if __name__ == "__main__":
    main()
