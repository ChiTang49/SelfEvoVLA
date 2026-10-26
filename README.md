# VLA-LoopLab

> 自进化 VLA 中的策略后训练、结果价值与验证器失配

VLA-LoopLab 是一个面向视觉—语言—动作模型的开源研究项目。项目关注的不只是如何搭建自进化链路，还包括：**价值模型与策略不同步时会发生什么，离线价值指标为何不能保证在线规划收益，以及测试时搜索在哪些条件下会失效。**

## 研究内容

- 在单张 32GB GPU 上完成 LIBERO-Long 的 RLOO/PPO 后训练。
- 实现有界动态采样、预算耗尽保护、零优势防护和采样诊断。
- 训练任务专属的轻量结果价值模型，对 K 个候选动作块进行排序。
- 使用仿真验证成功的轨迹完成一轮经验回流。
- 同时保留提升、无效和退化结果，分析策略、验证器与搜索之间的失配。

## 核心结果

每个任务使用一个策略训练 seed 和 50 个固定评估状态。主任务回流另使用两个评估 seed 复核。

| 任务 | SFT | 有界 RIPT | RIPT＋规划 | 价值留出 AUROC |
|---|---:|---:|---:|---:|
| 主任务 | 62% | 62% | 62% | 0.7893 |
| Task 5 | 66% | **74%** | 78% | 0.9094 |
| Task 6 | 56% | **88%** | 74% | 0.9519 |
| 宏平均 | 61.3% | **74.7%** | 71.3% | — |

主任务回流使用 21 条验证成功轨迹、843 个动作块微调 2 个 epoch：

| 模型与方式 | 两个评估 seed 的全量均值 | 留出均值 |
|---|---:|---:|
| 原 E3，K=1 | 67% | 65% |
| 回流，K=1 | 63% | 55% |
| 原 E3＋规划，K=4 | 66% | 55% |
| 回流＋规划，K=4 | **72%** | **75%** |

## 主要研究观察

1. **离线判别与在线选择不是同一个目标。** 较高的 AUROC/AUPRC 可以说明模型能区分成功与失败轨迹，但不能证明它能在同一状态下选出更好的动作块。
2. **冻结验证器可能与更新后的策略失配。** RIPT 改变了策略的数据分布，旧价值模型未必仍适合评价新策略。结果支持联合更新的研究方向，但不构成严格因果证明。
3. **更多候选并不必然更好。** K=1/4/8 没有单调收益。搜索也可能放大验证器误差，出现高分但低质量的动作。
4. **成功轨迹回流并非无条件有效。** 回流单独使用没有改善，但与规划组合后在两个评估 seed 上呈正向。单训练 seed 不能支撑普遍稳定性结论。
5. **有界采样首先是可控性改进。** 当前证据不足以宣称动态采样降低了总训练成本；其价值主要体现在重试上限、异常保护和诊断能力。

## 展示页面

仓库根目录就是静态网站，包含：

- `index.html`：项目动机与核心结果
- `method.html`：RIPT、结果价值规划与经验回流
- `results.html`：三任务、留出状态、消融与训练曲线
- `demo.html`：大尺寸行为对比视频
- `data.html`：CSV、失败案例与数据说明

GitHub Pages 地址：<https://chitang49.github.io/vla-looplab/>

本地预览：

```bash
python -m http.server 8000
```

然后访问 `http://localhost:8000`。修改页面内容后可运行：

```bash
python scripts/build_project_site.py
```

## 环境要求

- Linux、NVIDIA GPU，建议 32GB 显存
- Python 3.10
- PyTorch 2.2.0、TorchVision 0.17.0
- RIPT-VLA 的 QueST 分支及 LIBERO
- MuJoCo 2.3.7、EGL

参考版本：RIPT-VLA `440990e8864e12e4578b490ff6359e4f2c49ae3e`，LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`。

准备上游源码与配置：

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

按 `configs/ript_paths.yaml` 和 `configs/libero_config.yaml` 中的占位符设置本地数据路径。官方 SFT/RIPT 权重见 [RIPT_VLA checkpoints](https://huggingface.co/tanshh97/RIPT_VLA/tree/main/quest)。仓库不包含模型权重、LIBERO 数据集、Conda 环境或原始执行日志。

## 运行实验

以下命令以 Bash、单 GPU 和已准备的 LIBERO 环境为前提。`TASK_NAME` 必须使用上游任务列表中的完整名称。

### 评估 checkpoint

```bash
bash scripts/run_quest_eval.sh EXP_ID CHECKPOINT TASK_NAME 50 5 0
```

### RLOO 或有界 RIPT 训练

```bash
bash scripts/run_quest_train.sh EXP_ID TASK_NAME rloo 0
bash scripts/run_quest_train.sh EXP_ID TASK_NAME bript 0
```

先用 `smoke` 或 `bript-smoke` 验证单步训练与 checkpoint 重载，再启动完整训练。`RIPT_EXTRA_OVERRIDES` 可覆盖训练参数，`RIPT_DRY_RUN=1` 可检查覆盖项是否生效。

### 训练结果价值模型

```bash
PYTHONPATH=. python planning/train_value.py \
  --data-dirs evidence/SFT_ROLLOUTS evidence/RIPT_ROLLOUTS \
  --output evidence/VALUE_RUN/value_model.pth \
  --epochs 40 --batch-size 128 --seed 0
```

默认状态划分为 0–29 训练、30–39 验证、40–49 留出。

### 候选规划评估

```bash
NUM_CANDIDATES=4 VALUE_CHECKPOINT=evidence/VALUE_RUN/value_model.pth \
  bash scripts/run_quest_eval.sh EXP_ID CHECKPOINT TASK_NAME 50 1 0
```

`ROLLOUT_DIR` 可保存上下文与动作 token；`INIT_STATE_START` 支持从指定初始状态继续采集。

### 一轮经验回流

```bash
bash scripts/run_feedback_train.sh EXP_ID CHECKPOINT ROLLOUT_DIR TASK_NAME 2
```

仅使用训练状态范围内、经仿真确认成功的轨迹。不要将留出状态用于回流。

## 目录结构

```text
.
├── index.html / method.html / results.html / demo.html / data.html
├── assets/                 # 中文图、结果图、视频与样式
├── configs/                # 路径占位示例
├── evidence/               # 结果摘要、CSV 和训练曲线
├── patches/                # 上游适配与有界采样补丁
├── planning/               # 候选生成、价值模型、规划与回流
├── scripts/                # 训练、评估、分析与网站生成入口
└── .github/workflows/      # GitHub Pages 部署
```

## 结论边界

- 每个任务一个策略训练 seed，不声称跨训练 seed 稳定性。
- 结果价值模型是终端结果驱动的 Q 代理，不是精确 Bellman Q，也不是模型预测规划。
- 不包含 RoboTwin、真机、LIBERO-90 全量复现或 SOTA 声明。
- 规划在三任务留出状态上均未优于对应 RIPT；回流单独使用未稳定改善。
- Task 5 使用原 5 轮＋恢复 10 轮的两阶段训练，不宣称优化器状态无缝连续。

## 上游与许可证

本项目使用并修改了以下开源组件：

- [RIPT-VLA](https://github.com/Ariostgx/ript-vla)：QueST、RLOO/PPO 与动态采样基础实现。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)：仿真任务与评估环境。

RIPT-VLA、LIBERO 及本项目均采用 MIT License。上游版权与许可证保留在 `LICENSE` 中。本项目是独立研究扩展，不代表上游作者认可本项目的实现或结果。
