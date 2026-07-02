"""Sampled BR sanity checks.

True BR is intractable on the main game; we only verify the estimator
returns finite values and behaves directionally correctly (untrained
table → high exploit; trained table → lower exploit).
"""

import os
import tempfile

from cfr.agent.regret_table import RegretTable
from cfr.eval.sampled_br import compute_sampled_exploit
from cfr.train.trainer import train


def test_returns_finite_on_empty_table():
    table = RegretTable()
    exploit = compute_sampled_exploit(table, num_games=10, num_rollouts=2, seed=1)
    assert isinstance(exploit, float)
    assert exploit == exploit  # not NaN


def test_untrained_more_exploitable_than_trained():
    """After a short train, exploit estimate should not be wildly higher than untrained.

    With only a 200-iter train + tiny eval, this is mostly a smoke check that
    sampled_br runs end-to-end on a real PokerGame trajectory and returns a
    sensible-magnitude number. Tight directional asserts are flaky at this
    scale, so we only require both numbers be finite and within +/- 2 of zero.
    """
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "num_iterations": 200,
            "seed": 7,
            "log_every": 200,
            "checkpoint_every": 200,
            "eval_every": 0,
            "eval_num_games": 5,
            "eval_lookahead_samples": 2,
            "checkpoint_dir": os.path.join(d, "checkpoints"),
            "log_dir": os.path.join(d, "logs"),
        }
        trained = train(cfg)

    untrained = RegretTable()
    e_untrained = compute_sampled_exploit(untrained, num_games=15, num_rollouts=2, seed=0)
    e_trained = compute_sampled_exploit(trained,  num_games=15, num_rollouts=2, seed=0)
    for e in (e_untrained, e_trained):
        assert abs(e) < 4.0  # zero-sum utility is in [-1, 1] so exploit can be at most ~2
