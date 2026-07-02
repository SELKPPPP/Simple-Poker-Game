"""MCCFR training loop for the main PokerGame.

Per iteration:
    - Create a fresh PokerGame with a per-iter seed (outer RNG advances).
    - Pick traverser alternating by iter parity.
    - external_sampling walks the tree, mutating regret_table.

Periodically:
    - log_every:        print + append CSV row (iter, time, info_set_count, [exploit])
    - checkpoint_every: pickle the table
    - eval_every:       sampled BR exploit (slow; control via config)
"""

import csv
import os
import random
import time
from typing import Any, Callable, Dict, Optional

from ..agent.info_set import encode_info_set
from ..agent.mccfr import outcome_sampling
from ..agent.regret_table import RegretTable
from ..env.game import PokerGame
from ..eval.sampled_br import compute_sampled_exploit
from .checkpoint import get_git_hash, save_checkpoint


def _info_set_fn(game: PokerGame, player: int) -> str:
    return encode_info_set(game.observation(player))


def _ensure_log(log_dir: str, git_hash: str, start_iter: int) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"train_{git_hash}.csv")
    if start_iter == 0 or not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["iter", "elapsed_sec", "info_set_count", "sampled_exploit"]
            )
    return path


def train(
    config: Dict[str, Any],
    start_iter: int = 0,
    table: Optional[RegretTable] = None,
) -> RegretTable:
    rng = random.Random(config["seed"])
    # Advance the rng so resumed runs don't redo identical samples.
    for _ in range(start_iter):
        rng.randint(0, 2**31 - 1)

    if table is None:
        table = RegretTable()

    git_hash = get_git_hash()
    log_path = _ensure_log(config["log_dir"], git_hash, start_iter)
    t0 = time.time()

    num_iter = config["num_iterations"]
    log_every = config["log_every"]
    eval_every = config.get("eval_every", 0)
    ckpt_every = config["checkpoint_every"]

    for it in range(start_iter, num_iter):
        traverser = it % 2
        game = PokerGame(seed=rng.randint(0, 2**31 - 1))
        outcome_sampling(game, traverser, table, _info_set_fn, rng, weight=it + 1)
        done = it + 1

        do_log = done % log_every == 0
        do_eval = eval_every and done % eval_every == 0
        do_ckpt = done % ckpt_every == 0

        if do_log or do_eval or do_ckpt:
            elapsed = time.time() - t0
            info_count = len(table.info_sets())
            exploit = None
            if do_eval:
                exploit = compute_sampled_exploit(
                    table,
                    num_games=config["eval_num_games"],
                    num_rollouts=config["eval_lookahead_samples"],
                    seed=done,
                )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([done, f"{elapsed:.2f}", info_count, exploit])
            msg = f"iter={done} elapsed={elapsed:.1f}s info_sets={info_count}"
            if exploit is not None:
                msg += f" sampled_exploit={exploit:.4f}"
            print(msg)

            if do_ckpt:
                path = save_checkpoint(
                    table, done, config["seed"], config,
                    directory=config["checkpoint_dir"], git_hash=git_hash,
                )
                print(f"  checkpoint -> {path}")

    return table
