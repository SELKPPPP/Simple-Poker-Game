"""Cumulative regret + cumulative strategy storage for MCCFR.

Per info set:
    regrets[a]              = cumulative counterfactual regret (drives next-iter strategy)
    cumulative_strategy[a]  = sum of strategies used across iterations
                              normalized -> average strategy (the Nash-convergent quantity)

Strategy used by MCCFR at iteration t: regret matching on regrets.
Strategy used at inference time: normalized cumulative_strategy.
"""

from collections import defaultdict
from typing import Dict, Iterable, List


def _float_dict() -> Dict[int, float]:
    return defaultdict(float)


class RegretTable:
    def __init__(self):
        self._regrets: Dict[str, Dict[int, float]] = defaultdict(_float_dict)
        self._cumulative_strategy: Dict[str, Dict[int, float]] = defaultdict(_float_dict)

    def get_strategy(self, info_set: str, legal: Iterable[int]) -> Dict[int, float]:
        """Current-iter strategy via regret matching: P(a) ∝ max(0, regret(a))."""
        legal_list = list(legal)
        # Use .get() with empty fallback so a read never materialises a new entry.
        entry = self._regrets.get(info_set, {})
        positive = {a: max(0.0, entry.get(a, 0.0)) for a in legal_list}
        total = sum(positive.values())
        if total > 0:
            return {a: p / total for a, p in positive.items()}
        return {a: 1.0 / len(legal_list) for a in legal_list}

    def update_regret(self, info_set: str, action: int, delta: float) -> None:
        self._regrets[info_set][action] += delta

    def update_cumulative_strategy(self, info_set: str, strategy: Dict[int, float]) -> None:
        for a, p in strategy.items():
            self._cumulative_strategy[info_set][a] += p

    def average_strategy(self, info_set: str, legal: Iterable[int]) -> Dict[int, float]:
        legal_list = list(legal)
        entry = self._cumulative_strategy.get(info_set, {})
        cumul = {a: entry.get(a, 0.0) for a in legal_list}
        total = sum(cumul.values())
        if total > 0:
            return {a: c / total for a, c in cumul.items()}
        return {a: 1.0 / len(legal_list) for a in legal_list}

    def info_sets(self) -> List[str]:
        return list(self._cumulative_strategy.keys())
