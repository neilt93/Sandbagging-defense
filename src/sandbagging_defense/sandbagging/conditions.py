"""Phase 3: sandbagging system-prompt conditions.

Three conditions are loaded from configs/sandbag.yaml:
  - normal             : standard helpful assistant
  - sandbag_targeted   : underperform on capability tests, maintain plausibility
  - sandbag_general    : bland low-confidence answers across the board

Each condition is a (id, system_prompt) pair applied to a backend at
construction time. The Phase 3 driver instantiates one backend per
(model, condition) and runs the full benchmark suite under each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True, slots=True)
class Condition:
    id: str
    system_prompt: str


def load_conditions(config_path: Path) -> list[Condition]:
    cfg = OmegaConf.load(config_path)
    return [Condition(id=c.id, system_prompt=c.system_prompt) for c in cfg.conditions]
