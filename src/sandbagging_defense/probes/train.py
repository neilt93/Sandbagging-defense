"""Phase 4 entry point: train logistic-regression linear probes.

Splits by quadruplet_id to prevent leakage. Cantos uniform-output and
Levinstein-Herrmann transfer controls are run from this entry point.

Implementation lands in Phase 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 4 not yet implemented. Config: {args.config}")


if __name__ == "__main__":
    main()
