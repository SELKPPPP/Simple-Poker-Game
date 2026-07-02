# Improvement Changelog

> 项目改进方案的所有版本演进、设计决策、代码改动的完整记录。
>
> **新改动加在最上面**（最新在前）。每条记录包括：触发原因、改动内容、影响范围、相关文件。
>
> **添加新条目时请按以下模板**：
> ```
> ## YYYY-MM-DD — <主题简述>
> **触发**：为什么要改
> **改动**：具体改了什么
> **测试**：测试变化（pass / 新增 / 删除）
> **文件**：相关文件清单
> ```

---

## 2026-07-02 — v4 收敛优化落地：重训 1M + 纯化推理，CFR 全面超越 Q-learning

**触发**：执行 `收敛优化实现计划.md`（Task 6-8）。用户最终目标：新 CFR bot 以数据证明胜过原 Q-learning bot。

**改动**：
1. **v4 粗抽象 + Linear CFR**（commit `9cf8195`，设计见 `收敛优化设计.md`）：info sets 4.36M → 实测 31,245；OS 路径 regret/cum-strat 增量乘迭代权重 `t`。
2. **重训 1M iter**（~4.4h，产物 `cfr_9cf8195_*` / `cfr_dd2e7f5_1000000.pkl` + 对应 csv——中途提交过一次代码导致 hash 换名，数据分两个文件）。50K 验闸时闸门 1 字面未过（uniform 62.8% > 30%），但归因显示为短训假信号（按质量加权仅 5.3%，全是低访问尾部；uniform% 随迭代单调降），经用户拍板直投全量。终态 uniform 51.5%（旧 93.1%）。
3. **Q-learning 适配器**（commit `dd2e7f5`）：`cfr/eval/qlearning_agent.py` 让旧 bot 在新游戏出牌（fold 阶段恒 PLAY；每轮一次贪心换牌（合法子集 argmax）后 STOP；状态编码逐位复刻 `backend/utils_train.py::get_state`）。`cfr/scripts/eval_agents.py` 评测 CLI。
4. **纯化推理**（关键胜负手）：诊断发现采样推理在实战中 12.9% 的 fold 决策点乱弃牌（51.4% fold 节点近均匀，采样≈掷硬币）。`CFRAgent` 改为 **argmax（purification）+ 永不 fold**（fold 弱被支配，§4.5.5），agent 变为确定性、去掉 seed 参数。头对头从 −0.136 翻到 +0.096。

**结果**（1500 局/项，seed=42，`eval_agents` 可复现）：

| 对手 | CFR | Q-learning | Δ |
|---|---|---|---|
| random | +0.765 | +0.716 | +0.049 |
| always_call | +0.163 | +0.037 | +0.126 |
| tight | +0.441 | +0.399 | +0.042 |
| loose | −0.184 | −0.280 | +0.096 |
| heuristic | −0.060 | −0.111 | +0.051 |
| **头对头** | **+0.096**（822-678，胜率 54.8%，≈3.7σ） | — | — |

CFR 在全部 5 个 baseline 上优于 Q-learning，头对头统计显著胜出。旧 run 最差项 vs always_call 由 −0.640 转 +0.163。

**遗留**：exploit 曲线 300K 后在 ~1.55 平台波动（低于旧 1.69–1.86 但未继续降）；uniform 51.5% 未达 30% 目标（低访问尾部）。若需进一步提升：更粗 T3 抽象 / 降 ε / 加训。

**测试**：120/120 通过（新增 qlearning 适配器 7 项 + CFRAgent 纯化/never-fold 3 项；`test_play_human` 的 mock 输入改为按提示应答，不再依赖 AI 行为轨迹）。

**文件**：
- `cfr/agent/info_set.py`、`cfr/agent/mccfr.py`、`cfr/train/trainer.py`（v4 + Linear CFR）
- `cfr/eval/cfr_agent.py`（纯化 + never-fold 推理）
- `cfr/eval/qlearning_agent.py`、`cfr/scripts/eval_agents.py`（新增）
- `cfr/scripts/check_convergence.py`（新增）
- 对应测试；`收敛优化设计.md`、`收敛优化实现计划.md`

---

## 2026-05-24 — 1M-iter 训练评估 + 未收敛诊断

**触发**：长训练 10⁶ iter（4h 15min）跑完，需要评估 CFR 实际强度。

**结果**：vs rule panel 500 局/对手与未训练几乎无差异，最差情况 vs `always_call` 平均效用 -0.640（验证 §4.5.5：训练后仍以可观概率 fold），vs `tight` 从 +0.12 降到 -0.084。**训练未达到任何可用收敛水平**。

**根因诊断**（详见 `Improvement/训练未收敛诊断.md`）：
- 1M-iter checkpoint 中 **93.1% 多动作 info set 的 cum_strat 是完全均匀的**
- `average_strategy` 因此退化为 uniform 随机 → CFRAgent 推理等价于未训练
- 机理：4.36M info sets / 1M iter ≈ 平均 3 访问/info set（中位数=1），首次访问时 regrets=0 → sigma=uniform → IS 修正下单次访问的 increment ~10¹⁰ 被首次 uniform 完全主宰
- 算法本身**没有 bug**：Kuhn + external_sampling 仍能收敛；问题是 OS-MCCFR 在 PokerGame 当前 info set 抽象规模下方差爆炸

**未做改动**：按用户决策**保留现状**，不改算法、不重训。本次定位为「踩坑 + 写诊断」，输出价值是分析能力和工程证据，不是 agent 强度。

**文件**：
- `Improvement/训练未收敛诊断.md`（新增，~250 行）
- `Improvement/CHANGELOG.md`（本条目）
- `Improvement/项目知识库.md`（§4.5.6 引用诊断报告）

**测试**：未改代码，103/103 仍通过。

---

## 2026-05-22 — Phase 4 完成（rule panel + CFR 推理接口 + 对战循环 + 人机 CLI）

**触发**：按 plan §6 Phase 4 提供 CFR 训练成果的评估通路（vs rule baselines）和人机对战 demo。

**改动**：
- 新建 `cfr/eval/rule_agents.py`：5 个基线 agent
  - `RandomAgent`：legal_actions 均匀随机（最弱基线）
  - `AlwaysCallAgent`：永远 action=0（FOLD 阶段 PLAY、REDRAW 阶段 STOP）—— 利用 §4.5.5 fold-dominance 设计
  - `TightAgent`：hand_rank==0 时 fold，已成 pair 时 STOP，否则丢 junk
  - `LooseAgent`：永不 fold，hand_rank<3 时激进 redraw junk
  - `HeuristicAgent`：阈值 + 对手 face-up 强弱判断 + 跨 round 配额保留
  - 统一接口 `act(game, player) -> int`；只读 `game.observation(player)` 不窥探对手底牌
- 新建 `cfr/eval/cfr_agent.py`：CFR 推理 agent
  - 用 `average_strategy` 而非 `get_strategy`（前者才是 Nash-convergent 量）
  - `CFRAgent.from_checkpoint(path)` 一行加载训练结果
  - 空表自动退化为均匀随机（由 `RegretTable.average_strategy` 内部保证）
- 新建 `cfr/eval/play.py`：`play_match(agent_a, agent_b, num_games, seed)` 对战循环
  - 偶数局 a=seat0 / 奇数局 a=seat1 中和先手优势
  - 返回 `{a_wins, b_wins, ties, a_avg_utility, num_games}`
- 新建 `cfr/scripts/play_human.py`：终端人机对战 CLI
  - ASCII 牌面渲染（`2S` `KH` 风格，`[??]` 表示未亮牌）
  - 友好输入：`p`/`f` 折叠、`0,2,4` 或 `stop` redraw、无效输入自动重提示
  - 命令行 `python -m cfr.scripts.play_human --checkpoint X --seed Y --human-seat 0|1`
  - 无 checkpoint 时 AI 走均匀随机（方便调试 UI）
- 新建知识库 §4.5.5 "fold 仍是弱占优"洞察（先于代码，与 Phase 4 同 commit）

**测试**：
- 新增 4 个测试文件，共 14 个测试：
  - `test_rule_agents.py`：5 个 agent vs Random 对战合法性、AlwaysCall 不变量、Tight 真的 fold
  - `test_play.py`：胜负数之和=num_games、avg_utility ∈ [-1,1]、AlwaysCall vs Tight 期望非负（§4.5.5 验证）
  - `test_cfr_agent.py`：空表 legal-action、对战不崩、checkpoint 加载
  - `test_play_human.py`：mock input、card_str、redraw 输入解析、重提示、整局完成 + "GAME OVER" 输出
- 全套 103 个测试通过（Phase 0-3 的 89 + Phase 4 的 14）

**实现陷阱**：
- AlwaysCall vs Tight 测试一开始想用严格 `> 0`，但 BO3 + push 可能让 avg utility 接近 0，改用 `>= 0` 更稳
- play_human 用 `_make_input_feeder` 抽象 monkeypatch 输入，避免每个测试都写状态机
- CFR agent 写法上偏向 stateless 函数会更短，最终用类是为了 `from_checkpoint` 工厂方法 + 持有 rng 状态

**设计决定**：
- 走纯 CLI 路线，不动 Backend/Frontend（v3 plan §1 范围约束）
- 5 个 rule agents 故意包含"被支配策略"（AlwaysCall / Tight 极端），目的是让 vs-CFR 胜率成为 fold-dominance §4.5.5 的可验证指标

**文件**：
- 新增：`cfr/eval/rule_agents.py`、`cfr/eval/cfr_agent.py`、`cfr/eval/play.py`、`cfr/scripts/play_human.py`、`cfr/tests/test_rule_agents.py`、`cfr/tests/test_play.py`、`cfr/tests/test_cfr_agent.py`、`cfr/tests/test_play_human.py`
- 修改：`Improvement/项目知识库.md`（加 §4.5.5）、`Improvement/CHANGELOG.md`

---

## 2026-05-22 — Phase 3 完成（训练设施 + Outcome Sampling MCCFR + 采样 BR）

**触发**：按 plan §6 Phase 3 实现主游戏 PokerGame 的训练循环 + checkpoint + exploit 评估。开工后发现 External Sampling 在 PokerGame 上完全跑不动（REDRAW 32-action × 多 iter → 32^k 分支爆炸），单 iter 30s 内进不去 100k 递归。决定走 Outcome Sampling MCCFR 路线（大动作空间标准做法）。

**改动**：
- 新建 `cfr/train/__init__.py`、`cfr/train/trainer.py`：训练循环 + 日志 (CSV) + 周期 checkpoint
  - `train(config, start_iter=0, table=None)`：alternate-traverser 调用 outcome_sampling
  - `_info_set_fn(game, player) = encode_info_set(game.observation(player))` 把 PokerGame 接到 Phase 1 编码上
  - 触发条件：每 log_every iter 打 CSV 行 + stdout；每 eval_every iter 跑 sampled_br；每 checkpoint_every iter pickle 表
- 新建 `cfr/train/checkpoint.py`：pickle 存 `{regret_table, iter, seed, git_hash, config}`
  - 文件名 `cfr_{git_hash}_{iter}.pkl`（plan §7）
  - git hash 通过 `git rev-parse --short HEAD` 获取，无 git 时降级为 "nogit"
- 新建 `cfr/eval/sampled_br.py`：采样 BR exploit（主游戏专用）
  - 真 BR 不可行（10⁷ info set），用 1-ply lookahead Monte Carlo BR 作 lower bound
  - opp 用 σ_avg；p 每次决策枚举所有 legal action，每个用 K 次 rollout（双方都用 σ_avg）估 Q，取 argmax
  - `compute_sampled_exploit(table, num_games=200, num_rollouts=5) = BR_0 + BR_1`
  - 跟 Kuhn 的 brute-force BR (`exploitability.py`) 并存，各用各的
- 新建 `cfr/configs/default.yaml`：最小超参（num_iterations, seed, log/checkpoint/eval cadence, eval 参数, 路径）
- 新建 `cfr/scripts/__init__.py`、`cfr/scripts/train.py`：CLI driver
  - `python -m cfr.scripts.train [--config X] [--resume CKPT] [--num-iter N]`
  - resume 时读 ckpt 拿回 regret_table + start_iter + seed
- 修改 `cfr/agent/mccfr.py`：
  - 保留 `external_sampling`（Kuhn 用）
  - 新增 `outcome_sampling`：OS-MCCFR 按 OpenSpiel python 实现严格对齐
    - ε-exploration 采样（默认 0.6）
    - 关键 IS correction：`child_values[sampled] = child_value / sample_dist[sampled]`（少了这个会卡在 exploit≈0.33）
    - regret update：`cf_action_value × opp_reach / sample_reach`
    - cumul strategy update：`my_reach × σ[a] / sample_reach`
  - trainer 调 outcome_sampling 而非 external_sampling
- 修改 `cfr/agent/regret_table.py`：
  - defaultdict 默认 factory 从 lambda 改成模块级 `_float_dict()`（lambda 不能 pickle）
  - `get_strategy` / `average_strategy` 用 `.get(info_set, {})` 替代 `[info_set]`，避免读触发 defaultdict 创建（之前 eval rollout 一调用就把 info_sets 数从 12k 灌到 135k）
- 修改 `.gitignore`：加 `cfr/checkpoints/`、`cfr/logs/`
- 新增 `cfr/tests/test_checkpoint.py`、`test_trainer.py`、`test_sampled_br.py`（8 个测试）

**实现踩坑（修复记录）**：
1. **External Sampling 完全跑不动**：PokerGame REDRAW 阶段满 budget 32 个合法 action，traverser 枚举每个 + 后续 iter 32 个 + …，总分支 32^7 量级。单 iter 30s 进不去 100k 递归调用 → 切换 OS。
2. **OS 公式踩 3 个坑才对**：
   - 第一版 (-σ[b] for non-sampled)：exploit 不降反升，1M iter 后 J 总是 bet。
   - 改 -σ[a*]：exploit 卡在 0.5。
   - 加上 OpenSpiel 的 `child_value / sample_dist[sampled]` IS correction：终于收敛，1M iter exploit ≈ 0.01（α=0.139, 3α=0.42, Q@0 check, Q@1 call 0.357≈1/3 都对）
3. **pickle 失败**：RegretTable 用 `defaultdict(lambda: defaultdict(float))`，lambda 不可 pickle → 改模块级函数。
4. **eval 膨胀 info_set 数**：sampled_br 的 rollout 调用 `average_strategy(key, ...)`，触发 `self._cumulative_strategy[key]` defaultdict 创建空 entry。一次 eval 把表从 12k 灌到 135k → 改成 `.get(info_set, {})`。

**Phase 3 设计决策（vs plan §6）**：
- exploit 评估方式从"全 BR"（plan 原文）降级为"采样 BR lower bound"。理由：主游戏 info set 10⁷+ 量级，全 BR 不可行；LBR 实现 1-2 天；采样 BR 半天可交付。
- 长训练不在 Phase 3 范围。Phase 3 只交付训练设施 + smoke 验证；用户自己挂长训练（几小时到 1 天）。

**验证结果**：
- 89 个测试全过（旧 81 + 新 8）
- OS-MCCFR 在 Kuhn 上 1M iter exploit ≈ 0.006-0.015（与 Nash 解析解逐项匹配，α≈0.14）
- PokerGame 单 iter ~3ms（External Sampling 的 ∞ 改善）
- 5k iter smoke：info_sets 增长线性 (5k iter → 25k info sets)，checkpoint + resume 工作
- sampled_br 在 trained / untrained 表上都返回有限值 ∈ [-2, 2]

**文件**：
- 新增：`cfr/train/__init__.py`、`trainer.py`、`checkpoint.py`、`cfr/eval/sampled_br.py`、`cfr/configs/default.yaml`、`cfr/scripts/__init__.py`、`train.py`、`cfr/tests/test_trainer.py`、`test_checkpoint.py`、`test_sampled_br.py`
- 修改：`cfr/agent/mccfr.py`（加 outcome_sampling）、`cfr/agent/regret_table.py`（pickle + 读不创建）、`.gitignore`
- 文档：`项目知识库.md` §3 加 Outcome Sampling 说明 + §6 路线图 Phase 3 标 ✅

---

## 2026-05-21 — 更新 `项目知识库.md` §6（目录树 + Phase 状态 + 当前位置）

**触发**：Phase 1、2 完成后未同步 §6，目录树漏了 Phase 1/2 新增文件，Phase 状态停留在 Phase 0 完成时的快照。

**改动**（`项目知识库.md` §6）：
- §6.1 目录树补全 Phase 1/2 文件：`kuhn.py`、`info_set.py`、`regret_table.py`、`mccfr.py`、`eval/exploitability.py`、`test_info_set.py`、`test_kuhn.py`；加 Phase 3/4 待建说明
- §6.2 Phase 路线表更新状态：Phase 1/2 ✅ 完成（带测试数 + 关键指标）、Phase 3 ⏳ 下一步
- §6.3 当前位置改为 2026-05-21，反映"81/81 测试通过、MCCFR 在 Kuhn 上匹配 Nash 解析解"；新增"Phase 3 已知挑战"小节说明 PokerGame chance 处理两种方案 A/B

**测试**：无（纯文档）

**文件**：
- `Improvement/项目知识库.md` §6.1 / §6.2 / §6.3

---

## 2026-05-21 — Phase 2 完成（MCCFR + Kuhn 验证通过，exploit=0.0027）

**触发**：Phase 1 完成后，按 plan §6 Phase 2 实现 MCCFR core 并在 Kuhn Poker 上验证正确性。

**改动**：
- 新建 `cfr/env/kuhn.py`：标准 Kuhn Poker 游戏类
  - 3 张牌（J/Q/K）、2 玩家、5 种 terminal 历史
  - 接口与 `PokerGame` 对齐（is_terminal/current_player/legal_actions/apply/utility）
  - 支持 `hands` 显式传入（便于 BR 枚举）和 `seed` 随机发牌
- 新建 `cfr/agent/regret_table.py`：`RegretTable` 类
  - 双表分离存储：`regrets`（产生 current strategy）+ `cumulative_strategy`（产生 average strategy）
  - `get_strategy` 实现 regret matching（max(0, regret) / total，total=0 时均匀）
  - `average_strategy` 推理时用
- 新建 `cfr/agent/mccfr.py`：External Sampling MCCFR 核心
  - traverser 节点：枚举所有 action，更新 regret + cumulative strategy
  - 对手节点：按当前 strategy 采样 1 个 action
  - cumulative strategy **只在 traverser 自己的 info set 更新**（标准 ES-MCCFR）
  - `train_kuhn(N, seed)` 训练入口，iter 奇偶轮换 traverser
- 新建 `cfr/eval/__init__.py`、`cfr/eval/exploitability.py`：BR-based exploitability
  - **关键正确性点**：BR 策略必须按 info set 决定 action，不能在 (hands, history) 上独立 max
  - 实现：brute-force 枚举 $2^{|I|}$ 个纯策略指派（Kuhn 6 info sets → 64 candidates）
  - 公式：exploit = BR_0_value + BR_1_value（zero-sum 下 Nash 时为 0）
- 新建 `cfr/tests/test_kuhn.py`：17 个测试
  - 14 个 Kuhn 规则测试（5 种 terminal、zero-sum、illegal action、info set 逻辑）
  - 2 个 MCCFR 定性测试（K bets more than J、K calls more than J）
  - 1 个 Phase 2 验收门：10⁵ iter → exploit < 0.01

**实现踩坑（修复记录）**：
1. **Exploitability 初版被 BR 作弊污染**：原 `_br_value` 在 game tree 每个 (hands, history) 节点独立取 max。这等于 BR 玩家能看到对手底牌，给出 BR 上界（不是真实 BR）。结果 100k iter 后报 exploit=0.55（实际收敛良好）。
   - **修复**：改为 brute-force 枚举所有 info-set → action 指派，强制 "BR 是 info set 的函数" 约束
2. **状态拷贝方式**：用 `copy.deepcopy(game)` 在 apply 前拷贝。Kuhn 状态小（hands+history+terminal），deepcopy 在微秒级
3. **chance node**：Kuhn 发牌只在 KuhnGame.__init__ 时发生 → 训练循环外层 shuffle 后传 hands 给 KuhnGame，避免 chance node 处理

**验证结果**（10⁵ iter, seed=42）：
- exploit = **0.0027**（门槛 0.01）✓
- 训出策略与 Kuhn 解析解逐项匹配（α ≈ 0.237）：
  - P0 K first bet 0.727 ≈ 3α
  - P1 J after check bluff 0.336 ≈ 1/3
  - P0 Q after check-bet call 0.575 ≈ α + 1/3

**测试**：64 → 81（+17：Kuhn 测试），全 pass

**待办**：Phase 3 需要解决 PokerGame 的 chance 处理（每 round 都有 shuffle，不像 Kuhn 一次性）

**文件**：
- 新增：`cfr/env/kuhn.py`、`cfr/agent/regret_table.py`、`cfr/agent/mccfr.py`、`cfr/eval/__init__.py`、`cfr/eval/exploitability.py`、`cfr/tests/test_kuhn.py`
- 文档：`项目知识库.md` §5.5 加 exploitability 定义 + BR 计算正确性陷阱 + Phase 2 验证结果

---

## 2026-05-21 — Phase 1 实现完成（info_set.py + 26 个测试全过）

**触发**：方案敲定后实现 Phase 1。

**改动**：
- 新建 `cfr/agent/__init__.py`、`cfr/agent/info_set.py`
- `encode_info_set(obs)` 实现：7 元组骨架 + v2/v3 扩展 + draw_type，输出 canonical string key
- 关键函数：
  - `_hand_rank_and_top` —— 用 `evaluate_hand` 直接拿 hand_rank 和 top kicker
  - `_suited_mask` —— ≥3 同花的位置标 1（沿用 v1 思路），同时返回 dominant_suit 供对手对齐用
  - `_straight_features` —— 返回 (mask, draw_type)，**draw_type 用「distinct completing ranks」 判断**：
    - ≥2 个不同 rank 能补成顺子 → OPEN_ENDED
    - 仅 1 个 → GUTSHOT
    - 这种判法天然处理 wheel 边界（如 2-3-4-5 既能补 6 又能补 A → open-ended）
  - `_opp_face_up_features` —— 对手 face-up 编码为 sorted tuple of (rank, suit_aligned_with_my_dominant)
- 新建 `cfr/tests/test_info_set.py`：26 个测试覆盖
  - draw_type 7 个用例（含 wheel 边界、open-ended/gutshot 区分、made straight）
  - suited_mask 4 个用例（≥4 同花 / 3 同花 / below threshold / made flush）
  - determinism 2 个
  - canonicality 2 个（花色置换、不同 K 花色）
  - completeness 5 个（不同 phase / redraw / round_wins / redraw history / open vs gutshot）
  - opp face-up 3 个（alignment 影响 key、位置无关、None 处理）
  - hidden info 不泄漏 1 个
  - 与真实 PokerGame 集成 smoke test 2 个

**测试**：38 → 64（38 Phase 0 + 26 Phase 1），全 pass

**文件**：
- 新增 `cfr/agent/__init__.py`、`cfr/agent/info_set.py`、`cfr/tests/test_info_set.py`

---

## 2026-05-21 — Phase 1 Info Set 设计敲定（7 元组骨架 + v2/v3 扩展 + draw_type）

**触发**：开始 Phase 1 时发现原 plan §6 Phase 1 写的 abstraction `(hand_rank, top_3_kickers, suit_pattern)` 存在严重信息丢失：
- 不区分 open-ended 顺子缺 1 张（命中率 17%）vs gutshot 卡顺（命中率 8.5%），两者最优策略差很多
- 完全没有对手 face-up 信息（v2 不完美信息升级失效）
- 没有 fold history、iterative redraw 序列等 v2/v3 新增的关键信息

用户提议用 v1 时代 `model/poker_env.py::_get_state()` 的 7 元组（HandRank, TopCard, SuitedMask, StraightMask, ...）作骨架 —— 这套 hand-crafted 特征跟人类决策对齐、状态空间小、训练 tractable。

**改动**：
- **方案最终敲定**：7 元组骨架 + 必要扩展 + StraightMask 加 1 位 draw_type
  - 自己手牌特征：(hand_rank, top_card, suited_mask, straight_mask, **draw_type**)
  - 对手 face-up 特征（v2 扩展）：(face_up_ranks, face_up_suit_aligned)
    - 花色编码必须跟自己 dominant suit 对齐（同 / 异 2 值），否则 "对手追同花" 信号丢失
  - 资源 & 比分、phase、round_num（原 7 元组扩展）
  - 本 round 进行中：my_redraw_actions_seq, opp_redraw_counts_seq, ...（v3 iterative redraw 扩展）
  - 过往历史：fold_history + redraw_history（v2 + v3 扩展）
- **draw_type 4 取值**：0=none, 1=open-ended, 2=gutshot, 3=already-straight
- **状态空间估算**：~10⁵-10⁶（vs 完全无损方案 10⁷-10⁸）
- **方案文档** `model_improvement_plan.md` §6 Phase 1 重写
- **知识库** `项目知识库.md` §5.3 重写：加方案历史背景、draw_type 设计原理、花色对齐做法、完全无损方案的取舍

**待办**：实现 `cfr/agent/info_set.py` + `cfr/tests/test_info_set.py`

**测试**：无（纯方案文档变更）

**文件**：
- `Improvement/model_improvement_plan.md` §6 Phase 1
- `Improvement/项目知识库.md` §5.3

---

## 2026-05-20 — 修补 `项目知识库.md` 内部引用 & 补文档头

**触发**：上一条改动留了个待办 —— `项目知识库.md` 内部还在引用旧的单文件结构（`PROJECT_GUIDE.md`、`§0 规则 N`）。这次一次性修干净。

**改动**：
- `项目知识库.md` 顶部新增文档头（用途说明 + 配套文档清单 + 阅读顺序）—— 之前是空白直接进 §1
- §6.1 目录树：`PROJECT_GUIDE.md` 行替换为两行（`Project_guide1.md` + `项目知识库.md`）
- §6.3 当前位置：文档清单加入 `Project_guide1.md`、改 `PROJECT_GUIDE.md（本文件）` → `项目知识库.md（本文件）`
- 附录 B 路由表：`本文件 §0 规则 5/1` → `Project_guide1.md 规则 5/1`
- 全文 grep 确认无 `PROJECT_GUIDE` / `§0` / `第〇部分` 残留（CHANGELOG 自身的历史记录不算）

**测试**：无（纯文档）

**文件**：
- `Improvement/项目知识库.md`

---

## 2026-05-20 — PROJECT_GUIDE.md 拆成两份

**触发**：知识背景和工作规则放在一个文件里太杂、阅读路径不清晰。拆开让两类内容各司其职。

**改动**：
- 拆分为两份文档：
  - `Project_guide1.md` —— 只放 Agent 工作规则（6 条），新加配套文档索引
  - `项目知识库.md` —— 放原 §1-6 知识内容 + 附录
- `Project_guide1.md` 调整：
  - 头部说明改为"配套文档"清单（指向 项目知识库.md / model_improvement_plan.md / CHANGELOG.md）
  - 去掉「第〇部分」标题
  - 规则 2 改为「开始 Phase 前把三份文档过一遍」（原来只指向 plan）
  - 规则 6 补完正文：何时更新 `项目知识库.md`、与 CHANGELOG 的视角差异
- 删除原 `PROJECT_GUIDE.md`

**待办**：`项目知识库.md` 内部还有几处引用是按"单文件"写的（§6 提到 PROJECT_GUIDE.md、附录 B 提到"本文件 §0 规则 N"），需要后续修一下指向 `Project_guide1.md`。

**测试**：无（纯文档）

**文件**：
- 新增 `Improvement/Project_guide1.md`、`Improvement/项目知识库.md`
- 删除 `Improvement/PROJECT_GUIDE.md`

---

## 2026-05-20 — 新增 PROJECT_GUIDE.md（知识库 + Agent 工作规则）

**触发**：方案演进多轮、Phase 0 反复修改后，未来 session 看现有文档无法快速理解"为什么这么做"和"该做什么"。需要一个固定入口同时承载：① 零基础知识背景；② 工作规则。

**改动**：
- 新建 `Improvement/PROJECT_GUIDE.md`，结构：
  - §0 Agent 工作规则（5 条：CHANGELOG 更新、看 plan 才动手、CLAUDE.md 四准则、Phase 间停下来确认、范围限 cfr/）
  - §1 博弈论基础（完美 / 不完美信息、Nash、混合策略）
  - §2 强化学习基础（MDP、Q-learning、POMDP——Q-learning 失效原因）
  - §3 CFR 家族（regret matching、average strategy、MCCFR 三变体、Deep CFR 对比）
  - §4 游戏规则与设计权衡（含被支配测试、v1→v2→v3 推翻原因）
  - §5 工程概念（simultaneous-move 建模、info hiding 物理隔离、canonical encoding、self-play、Kuhn 验证）
  - §6 目录与 Phase 路线（带当前状态）
  - 附录 A 术语速查表
  - 附录 B 问题→文档路由表

**测试**：无（纯文档）

**文件**：
- 新增 `Improvement/PROJECT_GUIDE.md`

---

## 2026-05-20 — Phase 0 v3：迭代 redraw + STOP

**触发**：用户指出单次 redraw 有数学漏洞 ——
- 例：手牌 5,6,7,8,K，open-ended 4 张顺子，单次换 K 完成概率仅 8/47 ≈ 17%
- 留再多 redraw 在本 round 是浪费的，每 round 只有一发子弹
- 跨 round 攒弹药也没意义，因为下一 round 同样只能开一枪

**改动**：
- **游戏规则**：REDRAW phase 改成可迭代循环
  - 每 iter：双方同时 commit action（0..31）
  - `action == 0` 语义改为 **STOP**（本 round 退出 redraw）
  - `action ∈ 1..31` = bitmask 重画对应位置
  - loop 直到双方都 STOP
  - 一方 STOP 后另一方可继续（asymmetric play depth per round）
  - 预算耗尽 → 唯一合法 action 是 0 → 强制 STOP
- **新增策略空间**：追牌（一直换直到中）/ 见好就收 / iter 次数 leak 信息 / 跨 round 资源投资
- **Observation 字段变化**：
  - 移除 `my_redraw_count_this_round` / `opp_redraw_count_this_round`（单次计数）
  - 新增 `my_redraw_actions_this_round`: Tuple[int, ...] —— 自己的 bitmask 序列
  - 新增 `opp_redraw_counts_this_round`: Tuple[int, ...] —— 对手的 count 序列（committed only）
  - 新增 `my_redraw_done` / `opp_redraw_done`: bool
  - `redraw_history` 改为 Tuple[Tuple[int, ...], ...] —— 每 round 完整 iter 序列
  - 加 `my_fold_history` / `opp_fold_history`: Tuple[bool, ...]
- **方案文档**：§2.2 重写为迭代 redraw 规则；§2.3 加 3 条关键设计点解释

**测试**：33 → 38（pass 38/38）
- 新增：`test_zero_budget_only_stop_is_legal`、`test_redraw_action_hidden_from_opp_in_same_iter`、`test_my_pending_redraw_visible_to_self`、`test_iterative_chase_until_budget_exhausted`、`test_asymmetric_play_p2_keeps_going_after_p1_stops`
- 改写：约 60% 测试需要更新 action sequence

**文件**：
- `Improvement/model_improvement_plan.md` §2.2 / §2.3
- `cfr/env/game.py`（重写 REDRAW 逻辑：`_start_redraw_iter` / `_finalize_iter`）
- `cfr/env/observation.py`（重写字段）
- `cfr/tests/test_game.py`（重写 helpers + 新增 iter 测试）

---

## 2026-05-20 — Phase 0 v2：Fold 移到 redraw 之前

**触发**：用户指出原 fold 设计被严格支配 ——
- 旧设计：先 redraw 后 fold；fold 此时不消耗任何额外资源 vs show 是免费的、可能赢
- 任何理性 agent 都不会 fold，CFR 必然学出 P(fold) = 0
- Fold 在博弈论上毫无价值，等于没加这个动作

**改动**：
- **Phase 顺序**：DEAL → FOLD → REDRAW → SHOWDOWN
- **Fold 新经济性**：fold 发生在 redraw 之前 → 不消耗任何 redraw 配额 → 整 7 次配额全留给后续 round
- **新策略空间**：跨 round 资源博弈（起手烂就 fold 攒弹药）
- **代码**：Phase enum 顺序对调；apply() 中先 fold 后 redraw；fold 触发直接 resolve 跳过 redraw；新增 `fold_history` 字段

**测试**：25 → 33（pass 33/33）
- 新增：`test_fold_preserves_full_redraw_budget`、`test_one_fold_preserves_full_budget`、`test_fold_skips_redraw_phase`、`test_fold_history_records_fold_action`、`test_redraw_history_is_none_for_folded_rounds`
- 改写：所有 action sequence

**文件**：
- `Improvement/model_improvement_plan.md` §2.2 / §2.3
- `cfr/env/game.py`
- `cfr/tests/test_game.py`

---

## 2026-05-20 — Phase 0 v1：初版 game env 实现

**触发**：方案 v3 拍板，开始 Phase 0 编码。

**改动**：
- 新建 `cfr/` 目录树（env / tests）
- 从 `model/poker_rules.py` 复制到 `cfr/env/poker_rules.py`（不改）
- 实现 `cfr/env/observation.py`：Observation dataclass，隐藏信息物理上 None
- 实现 `cfr/env/game.py`：PokerGame，BO3 + 2 hole + 3 face-up + simultaneous-move 模拟
- 实现 `cfr/tests/test_game.py`：25 个测试覆盖信息隐藏、游戏规则、错误路径

**测试**：25/25 pass

**文件**：新增 `cfr/` 目录及其下所有文件

---

## 2026-05-20 — 方案 v3：本地化 + Fold

**触发**：用户问"能不能不动后端、前端，只在本地改 model"。仔细想确实没必要为了训 CFR 改 backend broadcast / DB schema —— 信息隔离在 env 层用 `observation()` API 强制就完美，比 backend 改造安全且简洁 100 倍。同时确认要加 fold 动作（之前推荐过的 Tier 1 A），让隐藏信息真正成为博弈点。

**改动**：
- 方案文档 v2 → v3（覆盖）
- 删除 `game_rules_changes.md`（内容合并进主文档 §2）
- 范围明确：只动 model 层，不动 backend / frontend / DB / 在线训练
- 加 fold action 到游戏规则
- 评测改成 rule panel + CLI 对战
- Phase 5（backend 集成）改为可选

**Phase 编号**：
- Phase 0：Python game env（含隐藏信息）+ observation API
- Phase 1：Info set encoding + unit test
- Phase 2：MCCFR core + Kuhn Poker 验证
- Phase 3：本游戏训练循环 + checkpoint
- Phase 4：Eval（rule panel + CLI 对战脚本）
- Phase 5（可选）：Backend 集成

**时间预算**：~2.5-3 周 → 1.5-2 周编码 + 1-2 天训练

**文件**：
- `Improvement/model_improvement_plan.md`（v3 覆盖 v2）
- 删除 `Improvement/game_rules_changes.md`

---

## 2026-05-20 — 方案 v2：CFR 路线确认 + 算法选 MCCFR

**触发**：用户接受"3-4 周时间预算 + 旧 q_table 作废"代价，选 CFR 路线。在 MCCFR 和 Deep CFR 之间，根据本游戏规模（10⁵-10⁶ info sets）确定走 **External Sampling MCCFR**：
- 表格能进内存，不需要 NN 泛化
- 算法可解释，可直接 print regret 表 debug
- Kuhn Poker 上有正确性金标准可验证
- Deep CFR 是 overkill

**改动**：
- 方案文档 v1 → v2
- 算法范式整体替换：Q-learning → MCCFR
- 游戏升级为不完美信息博弈（hidden hole cards）
- 包含：后端 broadcast 信息隔离改造 + DB schema 改动 + 完整全栈整改
- 新增独立文档 `game_rules_changes.md`

**文件**：
- `Improvement/model_improvement_plan.md`（v2 覆盖 v1）
- `Improvement/game_rules_changes.md`（新增，后来 v3 删除）

---

## 2026-05-20 — 方案 v1：Q-learning 改进方案（已废弃）

**触发**：用户要求基于 CLAUDE.md 给 model 部分提升建议，但当时还未引入隐藏信息。初版方案围绕 Q-learning 做工程化改进。

**改动**：
- 提出 4 个分阶段改进：
  - Phase 0：评测脚手架
  - Phase 1：状态信息补全（kicker / opp_remaining_redraws）
  - Phase 2：Self-play 替代固定 rule bot
  - Phase 3：Reward 重设计（potential-based shaping）
  - Phase 4：决策点（是否上 DQN）

**为什么作废**：
- 用户提出加新博弈点的需求 → 选 Tier 2 C（隐藏信息）
- 一旦引入隐藏信息，游戏变成不完美信息博弈，Q-learning 从理论上就不适用（POMDP / 混合策略问题）
- 整个方案需要换算法范式

**文件**：
- `Improvement/model_improvement_plan.md`（v1，被 v2 覆盖）

---

## 设计决策的关键经验

1. **每次"修 Phase 0"都是因为发现了被严格支配的 action**：
   - v1→v2：fold-after-redraw 被 show 支配
   - v2→v3：单次 redraw 在某些手牌下利用不了剩余预算
   - 教训：定 game rules 时要做"决策被支配测试" —— 任何 action 在某 state 下是否被另一个 action 严格优于？是 → 设计有洞

2. **本地化决策让范围收窄 70%**：避免了 backend broadcast / DB / frontend 三个工程坑

3. **不要怕推翻 Phase 0**：v1 → v2 → v3 三次迭代花了几小时，但比训完 CFR 才发现策略空间太窄强 100 倍
