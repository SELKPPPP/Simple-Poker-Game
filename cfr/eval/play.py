"""Head-to-head match runner for any two agents.

Both agents must expose act(game, player) -> int. The loop alternates which
agent gets seat 0 each game to neutralise first-player advantage.
"""

import random
from typing import Protocol

from ..env.game import PokerGame


class Agent(Protocol):
    name: str
    def act(self, game: PokerGame, player: int) -> int: ...


def play_match(agent_a: Agent, agent_b: Agent, num_games: int,
               seed: int = 0) -> dict:
    """Play num_games rounds; return summary stats from agent_a's perspective.

    Returns dict with: a_wins, b_wins, ties, a_avg_utility, num_games.
    """
    rng = random.Random(seed)
    a_wins = b_wins = ties = 0
    total_a_utility = 0.0

    for g_idx in range(num_games):
        game = PokerGame(seed=rng.randint(0, 2**31 - 1))
        # Alternate seats: even games -> a=seat0, odd games -> a=seat1
        a_seat = g_idx % 2
        seats = {a_seat: agent_a, 1 - a_seat: agent_b}

        while not game.is_terminal():
            p = game.current_player()
            action = seats[p].act(game, p)
            game.apply(action)

        u_a = game.utility(a_seat)
        total_a_utility += u_a
        if u_a > 0:
            a_wins += 1
        elif u_a < 0:
            b_wins += 1
        else:
            ties += 1

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_avg_utility": total_a_utility / num_games,
        "num_games": num_games,
    }
