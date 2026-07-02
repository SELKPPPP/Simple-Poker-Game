"""MCCFR variants: External Sampling (small action spaces) and Outcome Sampling
(large action spaces).

External Sampling — used for Kuhn (action space = 2):
    Walk the tree from current state. At traverser's nodes enumerate all
    actions, compute each subtree value, update regret = subtree_value - node_value.
    At opp's nodes sample one action from their current strategy.
    Cost per iter: branches at every traverser node.

Outcome Sampling — used for PokerGame (action space up to 32):
    Sample ONE trajectory root-to-leaf. At every node sample one action.
    At traverser nodes, regret update uses importance correction so the
    estimator is unbiased. Cost per iter: linear in game depth.
    Trade-off: per-iter variance higher, more iters to converge, but each
    iter is many orders of magnitude faster — overall wall time much less.

Game interface required (both variants):
    is_terminal() -> bool
    current_player() -> int
    legal_actions() -> List[int]
    apply(action) -> None  (mutates; we deepcopy before apply)
    utility(player) -> float
"""

import copy
import random
from typing import Callable, Dict

from ..env.kuhn import ALL_CARDS, KuhnGame
from .regret_table import RegretTable


def external_sampling(
    game,
    traverser: int,
    regret_table: RegretTable,
    info_set_fn: Callable,
    rng: random.Random,
) -> float:
    if game.is_terminal():
        return game.utility(traverser)

    current = game.current_player()
    info_set = info_set_fn(game, current)
    legal = game.legal_actions()
    strategy: Dict[int, float] = regret_table.get_strategy(info_set, legal)

    if current == traverser:
        action_values: Dict[int, float] = {}
        for a in legal:
            child = copy.deepcopy(game)
            child.apply(a)
            action_values[a] = external_sampling(child, traverser, regret_table, info_set_fn, rng)
        node_value = sum(strategy[a] * action_values[a] for a in legal)
        for a in legal:
            regret_table.update_regret(info_set, a, action_values[a] - node_value)
        regret_table.update_cumulative_strategy(info_set, strategy)
        return node_value

    # Opponent: sample one action
    actions = list(strategy.keys())
    weights = [strategy[a] for a in actions]
    sampled = rng.choices(actions, weights=weights, k=1)[0]
    child = copy.deepcopy(game)
    child.apply(sampled)
    return external_sampling(child, traverser, regret_table, info_set_fn, rng)


def outcome_sampling(
    game,
    traverser: int,
    regret_table: RegretTable,
    info_set_fn: Callable,
    rng: random.Random,
    epsilon: float = 0.6,
    weight: float = 1.0,
) -> float:
    """One OS-MCCFR episode from current game. Returns value_estimate at root
    (not directly meaningful at call site).

    weight: Linear CFR iteration weight (typically t = iteration index). Scales
    both regret and cumulative-strategy increments so late, regret-differentiated
    visits outweigh early IS-inflated uniform ones.
    """
    return _os_episode(
        game, traverser, regret_table, info_set_fn, rng,
        my_reach=1.0, opp_reach=1.0, sample_reach=1.0, epsilon=epsilon,
        weight=weight,
    )


def _os_episode(
    game, traverser: int, regret_table: RegretTable, info_set_fn: Callable,
    rng: random.Random, my_reach: float, opp_reach: float, sample_reach: float,
    epsilon: float, weight: float,
) -> float:
    """OS-MCCFR following OpenSpiel formulation.

    Returns: value_estimate = σ(sampled_a) × child_value at this node.
        Propagated upward (NOT divided by sample_reach). The IS correction
        happens only at the regret/strategy update step, dividing by sample_reach.
    """
    if game.is_terminal():
        return game.utility(traverser)

    current = game.current_player()
    info_set = info_set_fn(game, current)
    legal = game.legal_actions()
    sigma: Dict[int, float] = regret_table.get_strategy(info_set, legal)
    n = len(legal)

    # Sampling distribution: ε-explore at traverser, raw σ at opp.
    if current == traverser:
        sample_dist = {a: epsilon / n + (1.0 - epsilon) * sigma[a] for a in legal}
    else:
        sample_dist = sigma

    actions = list(sample_dist.keys())
    weights = [sample_dist[a] for a in actions]
    a_star = rng.choices(actions, weights=weights, k=1)[0]
    p_sample = sample_dist[a_star]

    if current == traverser:
        new_my = my_reach * sigma[a_star]
        new_opp = opp_reach
    else:
        new_my = my_reach
        new_opp = opp_reach * sigma[a_star]

    child = copy.deepcopy(game)
    child.apply(a_star)
    child_value = _os_episode(
        child, traverser, regret_table, info_set_fn, rng,
        my_reach=new_my, opp_reach=new_opp,
        sample_reach=sample_reach * p_sample, epsilon=epsilon,
        weight=weight,
    )

    # Baseline-corrected (vanilla baseline = 0) child_values:
    #   sampled: child_value / p_sample   (IS correction at THIS node)
    #   others:  0
    # value_estimate = Σ σ × child_values = σ[a*] × child_value / p_sample.
    sampled_corrected = child_value / p_sample
    value_estimate = sigma[a_star] * sampled_corrected

    if current == traverser:
        cf_value = value_estimate * opp_reach / sample_reach
        for a in legal:
            cf_action_value = (sampled_corrected if a == a_star else 0.0) \
                              * opp_reach / sample_reach
            regret_table.update_regret(info_set, a, weight * (cf_action_value - cf_value))
        increment = {a: weight * my_reach * sigma[a] / sample_reach for a in legal}
        regret_table.update_cumulative_strategy(info_set, increment)

    return value_estimate


def train_kuhn(num_iterations: int, seed: int = 42) -> RegretTable:
    """Train MCCFR on Kuhn Poker for given iterations. Alternates traverser."""
    rng = random.Random(seed)
    table = RegretTable()
    cards = list(ALL_CARDS)

    for it in range(num_iterations):
        rng.shuffle(cards)
        hands = (cards[0], cards[1])
        traverser = it % 2
        game = KuhnGame(hands=hands)
        external_sampling(
            game, traverser, table,
            info_set_fn=lambda g, p: g.info_set_key(p),
            rng=rng,
        )
    return table
