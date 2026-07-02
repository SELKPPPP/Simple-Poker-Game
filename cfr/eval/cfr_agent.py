"""CFR agent for inference: sample actions from average_strategy.

Average strategy is the Nash-convergent quantity, not regrets — that's what we
use at play time. The current-iter strategy (via regret matching) is only for
training.
"""

import random
from typing import Optional

from ..agent.info_set import encode_info_set
from ..agent.regret_table import RegretTable
from ..env.game import PokerGame
from ..train.checkpoint import load_checkpoint


class CFRAgent:
    """Picks actions by sampling from average_strategy at the current info set.

    Empty / unseen info sets fall back to uniform random over legal actions
    (handled inside RegretTable.average_strategy).
    """
    name = "cfr"

    def __init__(self, table: RegretTable, seed: int = 0):
        self._table = table
        self._rng = random.Random(seed)

    @classmethod
    def from_checkpoint(cls, path: str, seed: int = 0) -> "CFRAgent":
        payload = load_checkpoint(path)
        return cls(payload["regret_table"], seed=seed)

    def act(self, game: PokerGame, player: int) -> int:
        obs = game.observation(player)
        key = encode_info_set(obs)
        legal = game.legal_actions()
        strategy = self._table.average_strategy(key, legal)
        actions = list(strategy.keys())
        weights = [strategy[a] for a in actions]
        return self._rng.choices(actions, weights=weights, k=1)[0]
