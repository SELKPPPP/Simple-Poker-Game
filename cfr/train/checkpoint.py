"""Pickle-based checkpoint persistence for MCCFR training.

A checkpoint stores everything needed to resume training:
    regret_table : RegretTable (with internal defaultdicts intact)
    iter         : last completed iteration (resume starts at iter)
    seed         : original RNG seed
    git_hash     : commit at training time (for traceability)
    config       : the full hyperparam dict

File name: cfr_{git_hash}_{iter}.pkl  (plan §7)
"""

import os
import pickle
import subprocess
from typing import Any, Dict, Optional

from ..agent.regret_table import RegretTable


def get_git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def checkpoint_path(directory: str, git_hash: str, iteration: int) -> str:
    return os.path.join(directory, f"cfr_{git_hash}_{iteration}.pkl")


def save_checkpoint(
    table: RegretTable,
    iteration: int,
    seed: int,
    config: Dict[str, Any],
    directory: str,
    git_hash: Optional[str] = None,
) -> str:
    os.makedirs(directory, exist_ok=True)
    git_hash = git_hash or get_git_hash()
    path = checkpoint_path(directory, git_hash, iteration)
    payload = {
        "regret_table": table,
        "iter": iteration,
        "seed": seed,
        "git_hash": git_hash,
        "config": config,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
