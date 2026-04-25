"""Phase 1 entry point: generate Kolmogorov-stratified stimulus quadruplets.

Implementation lands in Phase 1. This module is a stub so `make stimuli`
resolves and the import surface is fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raise NotImplementedError(
        f"Phase 1 not yet implemented. Would write {args.n} quadruplets "
        f"(seed={args.seed}) to {args.out}."
    )


if __name__ == "__main__":
    main()
