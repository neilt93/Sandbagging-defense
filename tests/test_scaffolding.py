"""Phase 0 smoke test: scaffolding imports cleanly and entry points are wired."""

from __future__ import annotations

import importlib

import pytest

SUBMODULES = [
    "sandbagging_defense",
    "sandbagging_defense.stimuli",
    "sandbagging_defense.stimuli.generators",
    "sandbagging_defense.stimuli.complexity",
    "sandbagging_defense.stimuli.quadruplets",
    "sandbagging_defense.stimuli.music",
    "sandbagging_defense.elicitation",
    "sandbagging_defense.elicitation.harness",
    "sandbagging_defense.sandbagging",
    "sandbagging_defense.sandbagging.conditions",
    "sandbagging_defense.sandbagging.run",
    "sandbagging_defense.probes",
    "sandbagging_defense.probes.extract",
    "sandbagging_defense.probes.train",
    "sandbagging_defense.analysis",
    "sandbagging_defense.analysis.pilot",
    "sandbagging_defense.analysis.sandbag",
    "sandbagging_defense.analysis.figures",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_imports(name: str) -> None:
    importlib.import_module(name)


def test_version() -> None:
    import sandbagging_defense

    assert sandbagging_defense.__version__ == "0.1.0"
