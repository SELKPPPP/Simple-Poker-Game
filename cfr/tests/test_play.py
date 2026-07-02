"""Match runner sanity checks."""

from cfr.eval.play import play_match
from cfr.eval.rule_agents import (
    AlwaysCallAgent, RandomAgent, TightAgent,
)


def test_wins_and_ties_sum_to_num_games():
    result = play_match(RandomAgent(seed=1), RandomAgent(seed=2),
                        num_games=20, seed=42)
    assert result["a_wins"] + result["b_wins"] + result["ties"] == 20
    assert result["num_games"] == 20


def test_avg_utility_is_in_unit_interval():
    """Utility is in {-1, 0, +1} so the avg is in [-1, 1]."""
    result = play_match(RandomAgent(seed=1), AlwaysCallAgent(),
                        num_games=10, seed=7)
    assert -1.0 <= result["a_avg_utility"] <= 1.0


def test_alwayscall_beats_or_ties_tight_on_average():
    """Per §4.5.5: AlwaysCall punishes Tight's fold tendency. Over enough games,
    AlwaysCall should win more than Tight (Tight gives up rounds for free)."""
    result = play_match(AlwaysCallAgent(), TightAgent(),
                        num_games=100, seed=0)
    assert result["a_avg_utility"] >= 0.0, \
        f"AlwaysCall should not lose on average to Tight, got {result['a_avg_utility']}"
