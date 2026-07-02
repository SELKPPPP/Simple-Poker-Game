"""Convergence diagnostics for a checkpoint (or a freshly-trained table).

Usage:
    python -m cfr.scripts.check_convergence --checkpoint cfr/checkpoints/<f>.pkl

Reports:
  - info_set_count
  - uniform cum_strat %  (target < 30%; was 93.1% in the failed 1M run)
"""

import argparse

from cfr.train.checkpoint import load_checkpoint


def uniform_fraction(table) -> tuple:
    uniform = total = 0
    for cs in table._cumulative_strategy.values():
        if len(cs) <= 1:
            continue
        total += 1
        vs = list(cs.values())
        if max(vs) - min(vs) < 1e-3:
            uniform += 1
    return uniform, total


def report(table) -> None:
    n = len(table.info_sets())
    u, t = uniform_fraction(table)
    pct = (100.0 * u / t) if t else 0.0
    print(f"info_set_count      : {n:,}")
    print(f"multi-action isets  : {t:,}")
    print(f"uniform cum_strat   : {u:,}/{t:,} = {pct:.1f}%   (target < 30%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    table = load_checkpoint(args.checkpoint)["regret_table"]
    report(table)


if __name__ == "__main__":
    main()
