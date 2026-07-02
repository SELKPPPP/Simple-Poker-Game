# Model 改进方案 v3（MCCFR + 本地化 + 含 fold）

> v1（Q-learning 优化版）作废 —— 游戏升级为不完美信息博弈
> v2（CFR + 全栈改造）作废 —— 后端改造工程量大且与"agent 变强"目标无关
> **v3 = 本地化路线**：纯 Python 自包含，不动 backend / frontend / DB

---

## 1. 范围

### 在范围内
- 新建 `cfr/` 目录，纯 Python self-play
- 实现 5 张牌 draw poker（含 hole cards 隐藏信息 + fold action）
- MCCFR (External Sampling) 训练
- 评测：rule panel + CLI 人机对战

### **不在**范围内
- 后端 `boradcast` 改造
- DB schema 改动
- Frontend 任何改动
- 在线增量学习
- Backend 集成（Phase 5 标记为**可选**，本计划不算它）

旧 `backend/utils_train.py` 和 `/model/train` endpoint 在本计划下**不动也不删**，等 Phase 5（如果做）再处理。

---

## 2. 游戏规则（在本地 Python env 实现）

### 2.1 牌型

- 标准 52 张牌，每 round 每位玩家 5 张
- **索引 0, 1 = hole cards**（暗牌，仅自己可见）
- **索引 2, 3, 4 = face-up cards**（明牌，双方可见）
- BO3，全场共享 7 次 redraw 配额（不变）

### 2.2 每 round action 序列

为 CFR 实现简洁，使用**模拟 simultaneous → 实际 sequential with hidden info** 的标准建模方式。

**Fold phase 在 redraw 之前；redraw phase 可迭代直到 STOP**：

```
1. Deal: 双方各拿 5 张牌
2. Fold phase（simultaneous）：
   - P1 看完自己 5 张牌，选 ∈ {play, fold}
   - P2 选 ∈ {play, fold}，不知道 P1 选了什么
   - 揭示：双方知道对手 fold 或 play
3. 如果有任一玩家 fold → 跳过 redraw，直接 resolve
4. 如果双方都 play → 进入 Redraw phase 迭代循环：
   loop:
     如果 P1 still active:
       P1 选 action ∈ {0..31}
         action = 0      → STOP（P1 本 round 退出 redraw）
         action ∈ 1..31  → 用 bitmask redraw 这些位置（消耗对应预算）
     如果 P2 still active:
       P2 同上，不知道 P1 本 iter 的 action
     iter 揭示：双方看到对手本 iter 是 STOP 还是 redraw 了几张（不知道是哪几张）
   当双方都 STOP → 结束 redraw phase
5. Showdown: 翻 hole cards，按 poker_rules 比牌
```

### 2.3 关键设计点

#### Fold 必须在 redraw 之前

否则 fold 严格被 "不换牌 + showdown" 支配（showdown 免费、可能赢；fold 保证输）。

Fold 的真实价值：
- 起手牌烂 → fold → **整 7 次 redraw 配额全留给后续 round**
- 跨 round 资源博弈：0:1 落后了，先 fold 把弹药全攒到决胜局

#### Redraw 必须迭代

否则"差一张就成顺子/同花"这类局面下，留再多 redraw 配额都用不出去（5,6,7,8,K 只能换 K 一次，中或不中只看运气）。

迭代后 redraw 真正成为资源决策：
- 追牌：4-card 顺子/同花 → 一直换那 1 张直到中或没钱
- 见好就收：换 1 张得到对子 → STOP，别再浪
- bluff via redraw count：连续换很多次反而暴露"我前面没成功"
- 跨 round 资源管理：知道追牌一般要 3-5 attempts，决定每 round 投资多少

#### Action 0 = STOP（语义合并）

`action == 0`（bitmask = 00000）在新设计里意为 STOP（本 round 退出 redraw），不再是"keep all 然后进 fold"。

- 预算耗尽 → 唯一合法 action 是 0 → 强制 STOP
- 一个玩家 STOP 后另一个还可以继续 redraw（asymmetric play depth per round）

### 2.4 不改的规则

- `poker_rules.py`（牌型评估、比牌）完全不动
- BO3 胜负判定不变
- 7 次共享 redraw 配额不变

### 2.5 实现要点：CFR 模拟 simultaneous move

在 game tree 里把 simultaneous 拆成两个 sequential decision node，但 player 2 的 info set **不包含** player 1 的当前 action：

```python
class GameNode:
    def info_set(self, player: int) -> str:
        # 第二个决策的 player 看不到第一个决策的 action
        # 只看到本阶段开始时的公共信息
        ...
```

这是 OpenSpiel 处理 simultaneous game 的标准做法。

---

## 3. 算法：External Sampling MCCFR

参数选择和理由见对话历史。核心循环：

```python
def mccfr_iteration(node, traversing_player):
    if node.is_terminal():
        return node.utility(traversing_player)
    
    if node.is_chance():
        outcome = node.sample_chance()
        return mccfr_iteration(node.apply(outcome), traversing_player)
    
    info_set = node.info_set(node.current_player())
    strategy = regret_match(regret_table[info_set])
    
    if node.current_player() == traversing_player:
        # 自己：枚举所有 action
        action_values = {}
        for a in node.legal_actions():
            action_values[a] = mccfr_iteration(node.apply(a), traversing_player)
        
        node_value = sum(strategy[a] * action_values[a] for a in node.legal_actions())
        
        # update regret
        for a in node.legal_actions():
            regret_table[info_set][a] += (action_values[a] - node_value)
        
        # update cumulative strategy（最终答案来自这个表）
        for a in node.legal_actions():
            avg_strategy_table[info_set][a] += strategy[a]
        
        return node_value
    else:
        # 对手：sample 一个 action
        action = sample_from(strategy)
        return mccfr_iteration(node.apply(action), traversing_player)
```

**两个表必须严格分开**：
- `regret_table` —— 用于产生 current strategy
- `avg_strategy_table` —— 累积，最终推理时用这个的 normalized 值（这是新手最常错的点）

---

## 4. 成功标准

| Phase | 必须达到 |
|---|---|
| 1 | Info set unit test 全过：相同 observation → 相同 info set string |
| 2 | **Kuhn Poker 上 MCCFR exploitability < 0.01**（CFR 实现正确性金标准）|
| 3 | 本游戏训练 N iterations，exploitability 单调下降 |
| 4 | vs rule panel 胜率：对每个 baseline bot ≥ 55%；对 random bot ≥ 80% |

任一项不过 → 不进下一阶段。

---

## 5. 目录结构

```
cfr/
├── env/
│   ├── poker_rules.py        # 从 model/ 复制，不改
│   ├── game.py               # PokerGame：state / transitions / hidden info / fold
│   ├── observation.py        # 严格的 player-view observation API
│   └── kuhn.py               # Kuhn Poker 实现，用于验证 MCCFR
├── agent/
│   ├── mccfr.py              # 核心算法
│   ├── regret_table.py       # regret + cumulative strategy 存储
│   ├── strategy.py           # average strategy 计算 + 推理时采样
│   └── info_set.py           # canonical info set encoding
├── train/
│   ├── trainer.py            # 训练循环
│   └── checkpoint.py         # save/load (pickle + 版本号 + git hash)
├── eval/
│   ├── exploitability.py     # 跟 best response 比
│   ├── opponents.py          # rule-based 对手 panel
│   ├── evaluate.py
│   └── play_cli.py           # 人机 CLI 对战
├── tests/
│   ├── test_info_set.py      # ⚠️ 必须
│   ├── test_regret_match.py
│   ├── test_kuhn.py          # ⚠️ MCCFR 正确性金标准
│   └── test_game.py          # game tree / hidden info / fold 逻辑
├── configs/
│   └── default.yaml
└── scripts/
    ├── train.py
    └── eval.py
```

---

## 6. 分阶段方案

### Phase 0 — Python game env（1 天）

- 实现 `cfr/env/game.py`：5 张牌发牌、隐藏 hole、redraw、fold、resolve
- 实现 `cfr/env/observation.py`：严格的 player-view API（物理上拿不到对手 hole）
- 写 `tests/test_game.py`：
  - 发牌后 P1 通过 observation 拿到的 P2 hand 中 index 0/1 是 None
  - fold 后正确判负、剩余 redraw 不消耗
  - 双方都 fold = push

**验证**：测试全过。

### Phase 1 — Info Set Encoding（2 天，**CFR 的命门**）

- 实现 `cfr/agent/info_set.py`
- **方案**：以 v1 时代 `model/poker_env.py::_get_state()` 的 7 元组（HandRank, TopCard, SuitedMask, StraightMask, RemainingRedraws, PlayerWins, OpponentWins）作**骨架**，扩展加入 v2/v3 新增的信息。StraightMask 额外加 1 位 `draw_type` 区分 open-ended / gutshot。
- **Info set 完整字段**：
  - **自己手牌特征**（沿用 7 元组思路）：
    - `my_hand_rank` (0-9，由 `evaluate_hand` 算出)
    - `my_top_card` (0-12)
    - `my_suited_mask`（5-tuple，≥3 同花位置标 1）
    - `my_straight_mask`（5-tuple，≥3 连张位置标 1）
    - `my_draw_type` ∈ {0=none, 1=open-ended, 2=gutshot, 3=already-straight}（**v3 新增 1 位**）
  - **对手 face-up 特征**（v2 隐藏信息新增）：
    - `opp_face_up_ranks`（sorted 3-tuple of 0-12）
    - `opp_face_up_suit_aligned`（每张相对自己 dominant suit：same / other，3-tuple）
  - **资源 & 比分**：`my_remaining_redraws`, `opp_remaining_redraws`, `my_round_wins`, `opp_round_wins`
  - **游戏阶段**：`phase` ('fold'/'redraw'), `round_num`
  - **本 round 进行中**（v3 iterative redraw 新增）：
    - `my_redraw_actions_seq`（自己 bitmask 序列）
    - `opp_redraw_counts_seq`（对手 count 序列）
    - `my_redraw_done`, `opp_redraw_done`, `opp_fold_this_round`
  - **过往 round 历史**（v2 fold + v3 iter 新增）：
    - `my_fold_history`, `opp_fold_history`
    - `my_redraw_history`（自己 iter bitmask 序列）, `opp_redraw_history`（对手 count 序列）
- 写 `tests/test_info_set.py`：
  - **canonicality**：等价 hand（如 ♠A♥K vs ♥A♠K 在无 flush 潜力时）映射到同一 info set string
  - **determinism**：相同 observation 重复调用得到同一 string
  - **completeness**：关键策略点上不同 observation 不能映射到同一 info set
  - **draw_type 正确性**：open-ended `8,9,T,J,3` → draw_type=1；gutshot `8,9,J,Q,3` → draw_type=2
  - **隐藏信息不泄漏**：对手 hole cards 不进入编码（即使在 game.py 内部有数据，info set 不能含）

**验证**：unit test 全过。

**注**：原 plan 写的 `(hand_rank, top_3_kickers, suit_pattern)` abstraction 信息丢失严重（不区分 open-ended/gutshot、完全无对手信息），已被本节方案替代。详见 `CHANGELOG.md` 2026-05-20 Phase 1 设计敲定条目。

### Phase 2 — MCCFR + Kuhn 验证（3–4 天）

- 实现 `cfr/agent/mccfr.py`（核心循环 + regret matching）
- 实现 `cfr/env/kuhn.py`（标准 Kuhn Poker，3 张牌）
- 写 `tests/test_kuhn.py`：跑 10⁵ iterations，验证 exploitability < 0.01

**⚠️ Kuhn 不过禁止进 Phase 3**。这是 CFR 实现正确性的金标准，任何 implementation bug 都会让它不收敛。

**验证**：Kuhn exploitability 收敛。

### Phase 3 — 主游戏训练（2 天编码 + 训练时间）

- 实现 `cfr/train/trainer.py`：训练循环 + checkpoint + 自动 exploitability 评估
- 实现 `cfr/train/checkpoint.py`：seed 固定、config 绑定保存、文件名带 git hash
- 写 `configs/default.yaml`：所有超参显式声明
- 跑训练几小时到 1 天，画 exploitability 曲线

**验证**：exploitability 单调下降，最终落到合理水平（具体阈值看 game tree 大小，训练时观察）。

### Phase 4 — 评测（1–1.5 天）

- 实现 `cfr/eval/opponents.py`：5 种 rule bot（rule_bot / aggressive / conservative / random / greedy_low），加 fold 行为
- 实现 `cfr/eval/evaluate.py`：跑 vs 每个对手 5000 matches
- 实现 `cfr/eval/play_cli.py`：你在终端跟 agent 对战
- 实现 `cfr/eval/exploitability.py`：训练期间可调用

**验证**：成功标准里的胜率指标全达成 + CLI 对战主观感受 agent 不傻

### Phase 5 — Backend 集成（**可选，不在本计划必交范围**）

如果将来要部署：
- 在 `boradcast` 加信息隔离（先按 §7 风险表里的方式做）
- 替换 `backend/utils_agent.py` 推理逻辑：从 `argmax Q` 改成 `sample from average_strategy`
- 旧 `backend/utils_train.py` 整个废弃

---

## 7. 工程化要求（重要）

| 项 | 要求 |
|---|---|
| Seed | `random.seed` + `np.random.seed` 必须固定且 log |
| Config | 用 yaml，跟 checkpoint pickle 一起存 |
| Checkpoint 命名 | `cfr_{git_hash}_{iter}_{timestamp}.pkl` |
| Checkpoint 不入 git | 加 `.gitignore` |
| 训练日志 | CSV，每 N iter 一行（iter、exploitability、time）|
| Eval 命令化 | `python scripts/eval.py --ckpt xxx.pkl --metric {exploit, panel}` |
| Average strategy | 推理用 average strategy 的 normalized 值，**不是** last iteration current strategy |
| Git workflow | 主 branch 稳定；每 Phase 完成打 tag (`v3-phase1`, `v3-phase2`...) |

---

## 8. 明确不做的事

- ❌ Multi-threading / 分布式（先单核跑通）
- ❌ Deep CFR（你的游戏规模不需要）
- ❌ 抽象基类 / 设计模式（实现一遍后再决定要不要重构）
- ❌ DI / logging framework（print + yaml 够用）
- ❌ 接 DB / S3 / backend（Phase 5 才碰）
- ❌ 多 agent 并行开发（任务高度串行，并行无收益）

---

## 9. 时间预算

| Phase | 编码 | 训练 |
|---|---|---|
| 0 | 1 天 | — |
| 1 | 2 天 | — |
| 2 | 3–4 天 | Kuhn 验证 ~10 min |
| 3 | 2 天 | 数小时到 1 天 |
| 4 | 1–1.5 天 | — |

**总计 ≈ 1.5–2 周编码 + 1–2 天训练**。

---

## 10. 旧资产处理

- `model/q_table.pkl`：保留当 baseline，**不动**
- `model/agent.py / train.py / poker_env.py`：保留，**不动**（CFR 走 `cfr/` 新目录）
- `backend/`：**完全不动**
- `frontend/`：**完全不动**
- 已删除：`Improvement/game_rules_changes.md`（内容已合并进本文档 §2）
