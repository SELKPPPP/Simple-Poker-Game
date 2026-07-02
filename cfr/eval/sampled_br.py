"""Sampled best-response exploitability for the main PokerGame.

True BR is intractable: ~10^7 info sets, ~10^60 game-tree paths. We compute
a Monte-Carlo lower bound on BR_p_value:

  - opp plays sigma_avg (avg strategy from RegretTable).
  - p at each decision: enumerate legal actions; for each, roll out K games
    where BOTH players play sigma_avg from there, average the terminal utility.
    Pick the argmax action.
  - Average p's terminal utility over N games.

Because p uses sigma_avg in the rollouts (not deeper BR), this UNDERESTIMATES
true BR — it's "1-ply lookahead BR + sigma rollout". Good enough for tracking
training progress: if even this loose bound drops, sigma is becoming harder
to exploit.

exploit_sampled(sigma) = BR_0_sampled + BR_1_sampled
"""

import copy
import random
from typing import Callable, Dict, List

from ..agent.info_set import encode_info_set
from ..agent.regret_table import RegretTable
from ..env.game import PokerGame


def _strategy_at(game: PokerGame, player: int, table: RegretTable) -> Dict[int, float]:
    obs = game.observation(player)
    key = encode_info_set(obs)
    return table.average_strategy(key, game.legal_actions())


def _sample_action(strategy: Dict[int, float], rng: random.Random) -> int:
    actions = list(strategy.keys())
    weights = [strategy[a] for a in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


def _rollout_with_avg(game: PokerGame, player: int, table: RegretTable,
                      rng: random.Random) -> float:
    """Both players play sigma_avg until terminal. Return player's utility."""
    g = copy.deepcopy(game)
    while not g.is_terminal():
        cur = g.current_player()
        strat = _strategy_at(g, cur, table)
        a = _sample_action(strat, rng)
        g.apply(a)
    return g.utility(player)


def _greedy_br_action(game: PokerGame, player: int, table: RegretTable,
                      num_rollouts: int, rng: random.Random) -> int:
    """1-ply lookahead: pick action with highest avg rollout return."""
    best_q = -float("inf")
    best_a = None
    for a in game.legal_actions():
        total = 0.0
        for _ in range(num_rollouts):
            child = copy.deepcopy(game)
            child.apply(a)
            total += _rollout_with_avg(child, player, table, rng)
        q = total / num_rollouts
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def _br_value(player: int, table: RegretTable, num_games: int,
              num_rollouts: int, rng: random.Random) -> float:
    """Estimate BR_player_value(sigma_avg) by N games of (BR vs sigma_avg)."""
    total = 0.0
    for _ in range(num_games):
        g = PokerGame(seed=rng.randint(0, 2**31 - 1))
        while not g.is_terminal():
            cur = g.current_player()
            if cur == player:
                a = _greedy_br_action(g, player, table, num_rollouts, rng)
            else:
                a = _sample_action(_strategy_at(g, cur, table), rng)
            g.apply(a)
        total += g.utility(player)
    return total / num_games


def compute_sampled_exploit(table: RegretTable, num_games: int = 200,
                            num_rollouts: int = 5,
                            seed: int = 0) -> float:
    """Lower-bound exploit estimate. Higher = sigma_avg more exploitable."""
    rng = random.Random(seed)
    br_0 = _br_value(0, table, num_games, num_rollouts, rng)
    br_1 = _br_value(1, table, num_games, num_rollouts, rng)
    return br_0 + br_1
