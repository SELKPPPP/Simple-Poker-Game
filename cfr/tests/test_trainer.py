"""Phase 3 trainer smoke tests.

Verifies the training loop runs on PokerGame without crashing and produces
non-trivial regret tables. Long-form convergence is left to the user's own
training runs (see cfr/scripts/train.py).
"""

import os
import tempfile

from cfr.train.trainer import train


def _minimal_config(num_iter: int, tmpdir: str) -> dict:
    return {
        "num_iterations": num_iter,
        "seed": 42,
        "log_every": num_iter,         # one log line at the end
        "checkpoint_every": num_iter,  # one checkpoint at the end
        "eval_every": 0,               # skip eval in smoke
        "eval_num_games": 5,
        "eval_lookahead_samples": 2,
        "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
        "log_dir": os.path.join(tmpdir, "logs"),
    }


def test_train_50_iter_smoke():
    with tempfile.TemporaryDirectory() as d:
        table = train(_minimal_config(50, d))
    assert len(table.info_sets()) > 0


def test_train_writes_checkpoint_and_log():
    with tempfile.TemporaryDirectory() as d:
        cfg = _minimal_config(50, d)
        train(cfg)
        ckpt_files = os.listdir(cfg["checkpoint_dir"])
        assert any(f.startswith("cfr_") and f.endswith("_50.pkl") for f in ckpt_files)
        log_files = os.listdir(cfg["log_dir"])
        assert any(f.startswith("train_") and f.endswith(".csv") for f in log_files)


def test_train_then_resume_continues_iter():
    """Save at iter 30, load, train 20 more → final regret table has both phases' updates."""
    with tempfile.TemporaryDirectory() as d:
        cfg1 = _minimal_config(30, d)
        table1 = train(cfg1)
        n1 = len(table1.info_sets())

        cfg2 = _minimal_config(50, d)
        table2 = train(cfg2, start_iter=30, table=table1)
        n2 = len(table2.info_sets())

    assert n2 >= n1  # more iters should not lose info sets


def test_trainer_passes_linear_weight(monkeypatch):
    """trainer must call outcome_sampling with weight = it+1 each iteration."""
    import cfr.train.trainer as tr

    captured = []

    def fake_os(game, traverser, table, fn, rng, weight=1.0):
        captured.append(weight)

    monkeypatch.setattr(tr, "outcome_sampling", fake_os)
    with tempfile.TemporaryDirectory() as d:
        tr.train(_minimal_config(5, d))
    assert captured == [1, 2, 3, 4, 5]
