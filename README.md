# Blind Humanoid Stair Climbing via Proprioceptive Interaction Memory / 基于本体交互记忆的人形机器人盲爬楼梯

This repository contains ongoing research on **blind humanoid locomotion and stair climbing with the Unitree G1**. The current focus is to make a humanoid infer useful local stair geometry from its own physical interactions, preserve that information across footsteps, and adapt its gait **without RGB, depth images, elevation maps, or explicit stair dimensions at deployment**.

本仓库用于研究 **Unitree G1 的无外感知人形机器人运动与盲爬楼梯**。当前工作的核心目标是：让机器人从自身与环境的物理交互中提取局部楼梯几何信息，在跨步过程中保持这部分信息，并在部署时 **不依赖 RGB、深度图、高程图或显式楼梯尺寸** 的情况下自适应调整步态。

> **Interaction as Perception / 交互即感知**  
> Successful footfalls reveal feasible support locations; toe–riser contacts reveal local forward boundaries.  
> 成功落脚提供可行支撑位置，脚尖与立板的碰撞提供局部前向边界信息。

**Detect → Structure → Remember → Adapt**  
**检测交互 → 结构化几何 → 跨步记忆 → 自适应步态**

---

## Research Question / 研究问题

**Can a blind humanoid recover enough local stair geometry from sparse physical interactions to adapt its stepping online?**

**人形机器人能否仅依靠稀疏的本体交互信息，在线恢复足够的局部楼梯几何，并据此调整落脚与步距？**

The project studies this question through a deployable interaction-event pipeline, structured cross-step memory, a gated slow latent representation, and event-driven stride adaptation.

本项目围绕这一问题构建了可部署的交互事件检测、结构化跨步记忆、门控慢变量以及事件驱动的步距自适应机制。

---

## Method at a Glance / 方法概览

~~~text
Deployable proprioception
        │
        ├── Footfall / touchdown events
        ├── Toe–riser interaction events
        └── Forward kinematics
        │
        ▼
Structured interaction memory
(footprints, collision anchors, cross-step geometry)
        │
        ▼
Gated slow latent memory
        │
        ▼
Locomotion policy
        │
        ▼
Unitree G1
~~~

### 1. Sparse interaction events / 稀疏交互事件

Instead of keeping a very long raw observation history, the method preserves the few interaction events that are most informative for local stair geometry.

相比直接堆叠很长的原始观测历史，本方法重点保留少量但对楼梯几何最有信息量的交互事件。

- **Footfall / footprint:** evidence of a feasible support location.  
  **落脚 / 足印：** 提供可行支撑位置。
- **Toe–riser contact:** evidence of a local forward boundary.  
  **脚尖–立板碰撞：** 提供局部前向边界。
- **Forward kinematics:** reconstructs the spatial location of each event.  
  **正向运动学：** 用于恢复事件发生的位置。

The repository also contains training and evaluation tools for deployable footfall and toe–riser detectors using proprioceptive history.

仓库中同时包含基于本体感觉历史的落脚事件与脚尖–立板事件检测器训练、数据导出和评估工具。

### 2. Structured interaction memory / 结构化交互记忆

Recent footprints and toe–riser contacts are converted into cross-step geometric relations rather than stored only as isolated frames. The memory includes adjacent-foot and same-foot stride relations, collision-to-footprint geometry, trend statistics, and a safe-stride belief.

近期足印和碰撞点不会只作为孤立帧保存，而是被转换为跨步几何关系，包括相邻脚关系、同脚步距、碰撞点与落脚点之间的几何关系、趋势统计以及安全步距估计。

Key implementation: <code>src/mjlab/tasks/velocity/mdp/observations.py</code>

### 3. Gated slow latent memory / 门控慢变量记忆

The structured interaction summary is fused with stair-focused proprioception and encoded by an MLP + LSTM into a compact candidate latent. A state-dependent memory gate controls how quickly new information is written:

- **Normal:** regular adaptation.
- **Stair Write:** fast incorporation of newly detected stair evidence.
- **Stair Memory:** preserve the established stair context while allowing slower geometry refinement.

结构化交互信息与楼梯相关的本体观测共同输入 MLP + LSTM，生成紧凑的候选 latent。随后由状态相关的门控机制控制信息写入速度：

- **Normal：** 正常更新。
- **Stair Write：** 快速写入新出现的楼梯证据。
- **Stair Memory：** 保持已建立的楼梯状态，同时允许几何信息缓慢修正。

The memory update follows

$
z_t=(1-\alpha_t)z_{t-1}+\alpha_t z_t^{\mathrm{cand}}
$

where a larger \(\alpha_t\) writes new evidence faster and a smaller \(\alpha_t\) preserves existing memory.

核心实现：<code>src/mjlab/rl/slow_latent_model.py</code>

### 4. Event-driven stride adaptation / 事件驱动的步距自适应

The stair interaction logic follows a probe–backoff–lock process:

~~~text
First reliable collision
        ↓
Enter stair context / fast write
        ↓
Valid stair landing and stride bootstrap
        ↓
Progressive forward probing
        ↓
Second valid collision / upper-bound evidence
        ↓
Backoff
        ↓
Lock / hold the estimated safe stride region
~~~

The **state transitions and stride targets are event-driven**, while **phase-specific RL rewards train the policy to realize the desired motion**. The leg motion itself is not hard-coded.

其中，**状态切换与步距目标由交互事件驱动**，而 **分阶段 reward 用于训练策略真正实现对应的运动行为**；具体关节运动并不是手工编码的。

---

## Training Strategy / 训练策略

The locomotion policy is trained with PPO and frozen-teacher guidance. The privileged teacher may use simulation-only information such as depth or height information during training, while the deployed student is restricted to deployable proprioceptive signals.

运动策略采用 PPO 与 frozen-teacher guidance 联合训练。训练阶段的 privileged teacher 可以使用深度、高程等仿真特权信息，而部署阶段的 student 仅使用可部署的本体感觉信号。

The slow latent is trained jointly with the policy. Auxiliary heads provide semantic supervision for interaction events, stair state, stair geometry, and safe-stride estimation.

慢变量与策略联合训练，并通过交互事件、楼梯状态、楼梯几何和安全步距等辅助任务塑造 latent 表征。

Relevant files:

- <code>third_party/rsl_rl/rsl_rl/algorithms/ppo_teacher_kl.py</code> — PPO + teacher guidance
- <code>src/mjlab/rl/slow_latent_model.py</code> — gated slow latent actor
- <code>src/mjlab/tasks/velocity/config/g1/blind_rough_slow_latent_env_cfg.py</code> — task and interaction-memory configuration

---

## Baselines and Ablations / 基线与消融

The repository includes several alternative temporal/context models for controlled comparison with the proposed interaction-memory policy. The implementations are available; systematic final comparison runs are still being completed.

仓库已经实现多种时序建模与上下文估计基线，用于与交互记忆方法进行受控比较。目前这些模型的代码与训练配置已经完成，系统性的最终对比实验仍在继续。

| Model / 模型 | Role / 作用 | Implementation |
| --- | --- | --- |
| **Feed-forward MLP** | Standard blind locomotion baseline with stacked proprioception / 标准堆叠本体观测基线 | <code>main</code> |
| **LSTM** | Recurrent temporal-memory baseline / 循环时序记忆基线 | <code>lstm_teacher_policy</code> |
| **Causal Transformer** | Causal attention over fixed observation-action history / 对固定观测-动作历史进行因果注意力建模 | <code>lstm_teacher_policy</code> |
| **DreamWaQ-style context encoder** | History-based beta-VAE context code with velocity estimation and observation reconstruction / 基于历史的 beta-VAE 上下文编码、速度估计与观测重建 | <code>exp/dwaq-ablation-slowlatent</code> |
| **Explicit stair-state variants** | Compare learned interaction memory against explicit stair-state information / 与显式楼梯状态信息进行对照 | <code>lstm_teacher_policy</code> |
| **Safe-stride / semantic variants** | Isolate stride estimation and semantic-memory design choices / 分析安全步距和语义记忆设计 | <code>exp/safe-stride-v1-baseline</code>, <code>exp/semantic-v2</code> |

The causal Transformer implementation uses a fixed history with a causal mask and predicts actions from observation-action tokens. The DWAQ branch uses a DreamWaQ-style beta-VAE encoder whose context code contains a learned latent together with an estimated base velocity and is trained with velocity, reconstruction, and KL losses.

Transformer 基线使用带 causal mask 的固定历史序列，并从 observation-action token 中预测动作。DWAQ 分支采用 DreamWaQ 风格的 beta-VAE 上下文编码器，通过速度估计、观测重建和 KL 正则学习历史上下文表示。

---

## Evaluation / 评估

<code>scripts/velocity_eval/</code> contains fixed evaluation and analysis tools, including:

- flat and stair terrain evaluation
- goal-pyramid stair climbing
- stair-height transition evaluation
- swing-height / stair-height sweeps
- footprint and toe-event detector evaluation
- policy latent collection and clustering / PCA analysis
- landing support and collision statistics

<code>scripts/velocity_eval/</code> 提供固定评估与分析工具，包括：

- 平地与不同楼梯高度评估
- 目标金字塔楼梯攀爬
- 楼梯高度切换评估
- 摆脚高度与楼梯高度 sweep
- 落脚与碰撞事件检测器评估
- policy latent 采集、聚类与 PCA 分析
- 落脚支撑质量与碰撞统计

For details, see <code>scripts/velocity_eval/README.md</code>.

---

## Deployment / 部署

The repository includes a G1 deployment stack under <code>deploy_lstm/</code>:

- ONNX inference
- C++ runtime
- Unitree G1 FSM integration
- 29-DoF joint-position control
- persistent recurrent / slow-latent state support
- deployment configuration for blind, LSTM, and slow-latent policies

仓库在 <code>deploy_lstm/</code> 中包含 G1 部署链路，包括 ONNX 推理、C++ runtime、Unitree G1 FSM、29 自由度关节位置控制，以及 LSTM / slow-latent 状态管理。

The deployment infrastructure is implemented; real-robot validation of the current interaction-memory policy is ongoing.

当前部署框架已经完成，最新交互记忆策略的实机验证仍在继续。

---

## Current Status / 当前进展

**Implemented / 已完成**

- proprioceptive interaction-event pipeline
- structured footprint / toe-riser memory
- gated slow-latent architecture
- event-driven probe → backoff → lock mechanism
- detector training and evaluation tools
- MLP, LSTM, Causal Transformer, and DWAQ-style comparison implementations
- fixed evaluation pipelines and latent-analysis tools
- Unitree G1 ONNX/C++ deployment infrastructure

**Ongoing / 进行中**

- reward and motion-quality refinement
- robust probing and recovery behavior
- systematic baseline and ablation evaluation
- final quantitative paper experiments
- real-robot validation

---

## Branch Guide / 分支说明

| Branch | Purpose |
| --- | --- |
| **[exp/stage2c-sparse-events](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/exp/stage2c-sparse-events)** | Current interaction-memory / sparse-event research branch; 当前主要研究分支 |
| **[main](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/main)** | Main training, evaluation, Teacher-KL and slow-latent infrastructure |
| **[lstm_teacher_policy](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/lstm_teacher_policy)** | LSTM, Causal Transformer, and explicit stair-state baselines |
| **[exp/dwaq-ablation-slowlatent](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/exp/dwaq-ablation-slowlatent)** | DreamWaQ-style context ablation |
| **[perception-training](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/perception-training)** | Interaction-event perception training |
| **[agent/toe-riser-detector-v3](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/agent/toe-riser-detector-v3)** | Toe-riser detector development |
| **[exp/safe-stride-v1-baseline](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/exp/safe-stride-v1-baseline)** | Safe-stride baseline experiments |
| **[exp/semantic-v2](https://github.com/qiaomu-shen/mjlab_blind_perception/tree/exp/semantic-v2)** | Semantic-latent experiments |

---

## Repository Structure / 代码结构

~~~text
src/mjlab/
├── rl/
│   └── slow_latent_model.py          # gated slow-latent policy
└── tasks/velocity/
    ├── config/g1/                    # G1 task and RL configurations
    └── mdp/
        ├── observations.py           # interaction memory and deployable observations
        ├── temporal_stair_rewards.py # probe / backoff / lock reward shaping
        ├── stair_geometry.py         # stair geometry labels and utilities
        └── stair_sequence_logging.py # stair event diagnostics

scripts/velocity_eval/                # detector training, evaluation, analysis
third_party/rsl_rl/                   # local RSL-RL + Teacher-KL extensions
deploy_lstm/                          # ONNX/C++ Unitree G1 deployment
teacher_policies/                     # privileged teacher checkpoints
~~~

Branch-specific model implementations include the Causal Transformer in <code>lstm_teacher_policy</code> and the DWAQ model / algorithm in <code>exp/dwaq-ablation-slowlatent</code>.

---

## Setup

An NVIDIA GPU is recommended. The project uses <code>uv</code> for environment management.

~~~bash
git clone https://github.com/qiaomu-shen/mjlab_blind_perception.git
cd mjlab_blind_perception
uv sync --extra cu128
~~~

Basic commands:

~~~bash
uv run train --help
uv run play --help
~~~

---

## Training

Example: train the blind rough Teacher-KL policy.

~~~bash
uv run train Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard
~~~

Example: train the target-navigation slow-latent policy.

~~~bash
uv run train Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard
~~~

Training logs are stored under

~~~text
logs/rsl_rl/<experiment_name>/<task_id>/<run_name>
~~~

with the run configuration saved next to each checkpoint.

---

## Evaluation Example / 评估示例

~~~bash
uv run python scripts/velocity_eval/eval_policy_on_terrains.py \
  Mjlab-Velocity-Blind-Rough-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes-per-terrain 50 \
  --num-envs 50
~~~

Goal-pyramid evaluation:

~~~bash
uv run python scripts/velocity_eval/eval_policy_goal_pyramid.py \
  Mjlab-Velocity-Blind-Rough-TargetNavigation-SlowLatent-TeacherKL-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --episodes 50 \
  --num-envs 50
~~~

---

## Export / 导出

Policies can be exported for deployment with the repository export utility:

~~~bash
uv run python src/mjlab/scripts/export.py -c <checkpoint> -t <task-name>
~~~

The deployed student does not require the privileged teacher.

---

## Notes

This repository is an active research codebase. Experimental branches may evolve independently while the main interaction-memory method and evaluation protocol are still being refined.

本仓库仍处于持续研究与实验迭代阶段。不同实验分支可能独立演化，当前主要工作仍集中在交互记忆方法、系统评估和实机验证上。
