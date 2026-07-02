"""Linear CFR weighting on the Outcome Sampling path.

With identical rng seed + game, the sampled trajectory and every a* choice are
identical across runs (both start from empty tables → uniform sigma everywhere).
So the ONLY difference a weight W makes is a linear scale on every regret and
cumulative-strategy increment. We assert exactly that.
"""

import random

from cfr.agent.info_set import encode_info_set
from cfr.agent.mccfr import outcome_sampling
from cfr.agent.regret_table import RegretTable
from cfr.env.game import PokerGame


def _fn(g, p):
    return encode_info_set(g.observation(p))


def test_weight_scales_cumulative_strategy_and_regret_linearly():
    t1 = RegretTable()
    outcome_sampling(PokerGame(seed=7), 0, t1, _fn, random.Random(123), weight=1.0)

    t10 = RegretTable()
    outcome_sampling(PokerGame(seed=7), 0, t10, _fn, random.Random(123), weight=10.0)

    # identical trajectory → identical info-set keys touched
    assert set(t1.info_sets()) == set(t10.info_sets())
    assert len(t1.info_sets()) > 0

    for iset in t1.info_sets():
        cs1 = t1._cumulative_strategy[iset]
        cs10 = t10._cumulative_strategy[iset]
        for a in cs1:
            assert abs(cs10[a] - 10.0 * cs1[a]) <= 1e-6 * max(1.0, abs(cs10[a]))
        r1 = t1._regrets[iset]
        r10 = t10._regrets[iset]
        for a in r1:
            assert abs(r10[a] - 10.0 * r1[a]) <= 1e-6 * max(1.0, abs(r10[a]))


def test_weight_defaults_to_one():
    # default weight (omitted) must equal explicit weight=1.0
    t_def = RegretTable()
    outcome_sampling(PokerGame(seed=11), 0, t_def, _fn, random.Random(99))
    t_one = RegretTable()
    outcome_sampling(PokerGame(seed=11), 0, t_one, _fn, random.Random(99), weight=1.0)
    for iset in t_def.info_sets():
        assert t_def._cumulative_strategy[iset] == t_one._cumulative_strategy[iset]
