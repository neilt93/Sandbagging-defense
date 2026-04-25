"""Regression test: every config in configs/ loads and resolves cleanly.

Catches dangling cross-file interpolations and Hydra defaults pointing
at non-existent group files — bugs that would otherwise stay dormant
until the corresponding phase tries to load the config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_config_loads_and_resolves(path: Path) -> None:
    cfg = OmegaConf.load(path)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def test_pilot_reframings_path_exists() -> None:
    pilot = OmegaConf.load(CONFIG_DIR / "pilot.yaml")
    referenced = Path(pilot.reframings_path)
    assert referenced.exists(), f"pilot.reframings_path missing: {referenced}"


def test_sandbag_pilot_config_path_exists() -> None:
    sandbag = OmegaConf.load(CONFIG_DIR / "sandbag.yaml")
    referenced = Path(sandbag.pilot_config_path)
    assert referenced.exists(), f"sandbag.pilot_config_path missing: {referenced}"


def test_probes_sandbag_config_path_exists() -> None:
    probes = OmegaConf.load(CONFIG_DIR / "probes.yaml")
    referenced = Path(probes.sandbag_config_path)
    assert referenced.exists(), f"probes.sandbag_config_path missing: {referenced}"


def test_reframings_have_expected_templates() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "reframings.yaml")
    ids = {r.id for r in cfg.reframings}
    assert ids == {"structure", "predictability", "continuation_interest"}
    for r in cfg.reframings:
        assert "{a}" in r.template and "{b}" in r.template


def test_sandbag_has_three_conditions() -> None:
    cfg = OmegaConf.load(CONFIG_DIR / "sandbag.yaml")
    ids = [c.id for c in cfg.conditions]
    assert ids == ["normal", "sandbag_targeted", "sandbag_general"]
