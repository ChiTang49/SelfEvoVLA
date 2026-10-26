# SelfEvoVLA

自进化 VLA 中的策略后训练、结果价值规划与验证经验回流。项目基于 QueST/RIPT-VLA 和 LIBERO，在单张 32GB GPU 上运行。

## 方法

- RLOO/PPO 后训练与有界动态采样。
- 轻量结果价值模型，对 K 个候选动作块排序。
- 使用仿真验证成功的轨迹进行一轮经验回流。
- 保留全量、留出和负结果，分析规划收益与验证器失配。

## 结果

每个任务使用一个策略训练 seed 和 50 个固定评估状态。

| 任务 | SFT | 有界 RIPT | RIPT＋规划 | 价值留出 AUROC |
|---|---:|---:|---:|---:|
| 主任务 | 62% | 62% | 62% | 0.7893 |
| Task 5 | 66% | **74%** | 78% | 0.9094 |
| Task 6 | 56% | **88%** | 74% | 0.9519 |
| 宏平均 | 61.3% | **74.7%** | 71.3% | — |

主任务回流使用 21 条成功轨迹、843 个动作块。两个评估 seed 下，原规划与回流规划的全量均值分别为 66% 和 72%。

## 安装

环境：Linux、Python 3.10、PyTorch 2.2.0、TorchVision 0.17.0、MuJoCo 2.3.7、EGL。

参考版本：RIPT-VLA `440990e`，LIBERO `8f1084e`。

```bash
mkdir -p sources checkpoints/quest
git clone https://github.com/Ariostgx/ript-vla.git sources/ript-vla
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git sources/LIBERO

cp patches/eval_quest.py sources/ript-vla/eval_quest.py
cp patches/train_ript.py sources/ript-vla/train_ript.py
cp patches/libero_runner.py sources/ript-vla/ript/env_runner/libero_runner.py
cp patches/rl_optimizer.py sources/ript-vla/ript/algos/rl_optimizers/rl_optimizer.py
cp patches/rollout_generator.py sources/ript-vla/ript/algos/rl_optimizers/rollout_generator.py
cp patches/rl_optimizers_init.py sources/ript-vla/ript/algos/rl_optimizers/__init__.py
```

按 `configs/` 中的示例设置数据路径。基础权重见 [RIPT_VLA checkpoints](https://huggingface.co/tanshh97/RIPT_VLA/tree/main/quest)。仓库不包含模型权重和数据集。

## 运行

`TASK_NAME` 使用上游任务列表中的完整名称。

```bash
# 评估 checkpoint
bash scripts/run_quest_eval.sh EXP_ID CHECKPOINT TASK_NAME 50 5 0

# RLOO / 有界 RIPT
bash scripts/run_quest_train.sh EXP_ID TASK_NAME rloo 0
bash scripts/run_quest_train.sh EXP_ID TASK_NAME bript 0

# 训练结果价值模型
PYTHONPATH=. python planning/train_value.py \
  --data-dirs evidence/SFT_ROLLOUTS evidence/RIPT_ROLLOUTS \
  --output evidence/VALUE_RUN/value_model.pth \
  --epochs 40 --batch-size 128 --seed 0

# 候选规划评估
NUM_CANDIDATES=4 VALUE_CHECKPOINT=evidence/VALUE_RUN/value_model.pth \
  bash scripts/run_quest_eval.sh EXP_ID CHECKPOINT TASK_NAME 50 1 0

# 一轮经验回流
bash scripts/run_feedback_train.sh EXP_ID CHECKPOINT ROLLOUT_DIR TASK_NAME 2
```

建议先运行 `smoke` 或 `bript-smoke`，确认 checkpoint 可保存和重载。参数覆盖使用 `RIPT_EXTRA_OVERRIDES`，配置检查使用 `RIPT_DRY_RUN=1`。

## 展示页面

GitHub Pages：<https://chitang49.github.io/SelfEvoVLA/>

- `index.html`：项目概览
- `method.html`：方法
- `results.html`：实验结果
- `demo.html`：行为对比视频
- `data.html`：CSV 与失败案例

本地预览：

```bash
python -m http.server 8000
```

## 许可证

MIT License，见 `LICENSE`。
