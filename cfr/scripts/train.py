"""CLI driver for MCCFR training on PokerGame.

    python -m cfr.scripts.train                              # default config
    python -m cfr.scripts.train --config cfr/configs/x.yaml
    python -m cfr.scripts.train --resume cfr/checkpoints/cfr_abc_5000.pkl
    python -m cfr.scripts.train --num-iter 1000              # override
"""

import argparse
import os
import sys

import yaml

# Allow `python -m cfr.scripts.train` and `python cfr/scripts/train.py` both.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from cfr.train.checkpoint import load_checkpoint
from cfr.train.trainer import train


DEFAULT_CONFIG = "cfr/configs/default.yaml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--resume", default=None,
                   help="path to checkpoint .pkl to resume from")
    p.add_argument("--num-iter", type=int, default=None,
                   help="override num_iterations from config")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.num_iter is not None:
        config["num_iterations"] = args.num_iter

    start_iter = 0
    table = None
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        table = ckpt["regret_table"]
        start_iter = ckpt["iter"]
        # Honor original seed so the rng schedule is consistent on resume.
        config["seed"] = ckpt["seed"]
        print(f"resumed from iter={start_iter} ({args.resume})")

    train(config, start_iter=start_iter, table=table)


if __name__ == "__main__":
    main()
