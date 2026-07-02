"""CFR agent for inference: purified argmax over average_strategy + never-fold.

Average strategy is the Nash-convergent quantity, not regrets — that's what we
use at play time. The current-iter strategy (via regret matching) is only for
training.

Two inference-time hardenings over naive sampling (both validated on the 1M v4
checkpoint, see Improvement/CHANGELOG.md):

- Purification: play argmax of the average strategy instead of sampling it.
  ~half the info sets are low-visit tail whose average strategy is still near
  uniform; sampling there injects random blunders (e.g. folding ~50% of the
  time). Argmax also breaks exact ties toward the lowest action, which is
  PLAY in the FOLD phase and STOP in the REDRAW phase — safe defaults.
- Never-fold: fold is weakly dominated by play+STOP in this game
  (项目知识库 §4.5.5), so the FOLD phase is answered with PLAY outright
  instead of trusting under-trained fold-node strategies.

Head-to-head vs the legacy Q-learning bot this moved -0.136 (sampled) to
+0.096 (n=1500 games).
"""

from ..agent.info_set import encode_info_set
from ..agent.regret_table import RegretTable
from ..env.game import PokerGame, Phase, FOLD_ACTION_PLAY
from ..train.checkpoint import load_checkpoint


class CFRAgent:
    """Deterministic purified inference over a trained regret table.

    Empty / unseen info sets fall back to uniform (inside
    RegretTable.average_strategy), where argmax degrades to the lowest legal
    action — PLAY / STOP.
    """
    name = "cfr"

    def __init__(self, table: RegretTable):
        self._table = table

    @classmethod
    def from_checkpoint(cls, path: str) -> "CFRAgent":
        payload = load_checkpoint(path)
        return cls(payload["regret_table"])

    def act(self, game: PokerGame, player: int) -> int:
        if game.phase == Phase.FOLD:
            return FOLD_ACTION_PLAY
        obs = game.observation(player)
        key = encode_info_set(obs)
        strategy = self._table.average_strategy(key, game.legal_actions())
        return max(strategy, key=strategy.get)  # first max wins ties
