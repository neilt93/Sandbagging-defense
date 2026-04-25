"""Phase 2 entry point: vLLM pairwise preference harness.

Implementation lands in Phase 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Phase 2 not yet implemented. Config: {args.config}")


if __name__ == "__main__":
    main()
