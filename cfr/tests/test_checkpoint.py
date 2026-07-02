"""Checkpoint save / load roundtrip."""

import os
import tempfile

from cfr.agent.regret_table import RegretTable
from cfr.train.checkpoint import load_checkpoint, save_checkpoint


def _make_table() -> RegretTable:
    t = RegretTable()
    t.update_regret("a", 0, 0.5)
    t.update_regret("a", 1, -0.2)
    t.update_cumulative_strategy("a", {0: 0.7, 1: 0.3})
    t.update_cumulative_strategy("b", {0: 1.0, 1: 0.0})
    return t


def test_save_creates_file_with_expected_name():
    table = _make_table()
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(table, iteration=42, seed=7,
                               config={"k": "v"}, directory=d,
                               git_hash="abc1234")
        assert os.path.basename(path) == "cfr_abc1234_42.pkl"
        assert os.path.exists(path)


def test_load_returns_equivalent_payload():
    table = _make_table()
    cfg = {"num_iterations": 100, "seed": 7}
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(table, iteration=42, seed=7,
                               config=cfg, directory=d, git_hash="xyz")
        loaded = load_checkpoint(path)
    assert loaded["iter"] == 42
    assert loaded["seed"] == 7
    assert loaded["git_hash"] == "xyz"
    assert loaded["config"] == cfg
    t2 = loaded["regret_table"]
    # Strategy from cumul matches
    avg = t2.average_strategy("a", [0, 1])
    assert abs(avg[0] - 0.7) < 1e-9
    assert abs(avg[1] - 0.3) < 1e-9
    # Regrets preserved -> current strategy via regret matching
    strat = t2.get_strategy("a", [0, 1])
    # only action 0 has positive regret -> P(0)=1
    assert strat[0] == 1.0
    assert strat[1] == 0.0


def test_load_after_save_preserves_info_sets():
    table = _make_table()
    with tempfile.TemporaryDirectory() as d:
        path = save_checkpoint(table, iteration=1, seed=0,
                               config={}, directory=d, git_hash="h")
        loaded = load_checkpoint(path)
    assert set(loaded["regret_table"].info_sets()) == {"a", "b"}
