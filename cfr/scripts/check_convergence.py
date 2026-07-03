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
    """Count multi-action info sets whose NORMALIZED average strategy is
    within 1pp of uniform (zero-mass sets play uniform too).

    Normalizing first keeps the check invariant to the raw cum-strat scale,
    which grew from O(t) to O(t^2) when Linear CFR weighting landed. Measured
    impact vs the old raw 1e-3 check: 93.1%->93.3% (old 1M run),
    51.5%->52.0% (v4 1M run) — conclusions unchanged, definition now sound.
    """
    uniform = total = 0
    for cs in table._cumulative_strategy.values():
        if len(cs) <= 1:
            continue
        total += 1
        vs = list(cs.values())
        s = sum(vs)
        if s <= 0:
            uniform += 1
            continue
        p = [v / s for v in vs]
        if max(p) - min(p) < 0.01:
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
