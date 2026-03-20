# SpaceStation-MRTA

**基于约束预演与注意力机制的空间站异构多机器人任务分配系统**

> 本项目基于论文 [Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning](https://arxiv.org/abs/2409.00145) (RAL 2024, Weiheng Dai et al.) 的开源代码进行二次开发，面向**空间站舱内异构微型机器人集群**的多技能累加型（MA-AT）协同任务分配场景，针对原框架在能力维度、任务类型、时空约束和奖励机制等方面进行了系统性改造。

## 1. 项目概述

### 1.1 问题背景

空间站常态化运营中，异构机器人集群（巡检、操作、感知）需要协同完成大量多技能累加型运维任务。这类问题具有以下核心挑战：

- **多技能累加（MA-AT）**：复杂任务需要多种异构机器人同时在场、能力叠加方可启动，如 `[0, 1, 1]` 表示需要 1 个操作机器人 + 1 个感知机器人协作
- **强时空耦合**：不同舱段（Zone）存在物理准入约束与时效要求，任务具有严格截止时间（Deadline）
- **协同死锁风险**：独立决策的机器人易因互相等待而陷入系统级死锁

### 1.2 方法简介

本系统采用**基于注意力机制的去中心化强化学习框架**，通过以下核心技术解决上述问题：

| 技术模块 | 核心机制 |
|---------|---------|
| 双流异构编码器 | 并行编码智能体图与任务图，通过 Cross-Attention 实现供需特征交互 |
| 顺序条件化决策 | 建立决策依赖链，后序机器人可观测前序动作更新后的残差需求，实现隐式意图对齐 |
| 约束预演机制 (CFM) | 训练时通过前瞻推演引导决策顺序，缓解死锁并提升探索效率 |
| 动态动作空间裁剪 | 将物理可行性约束（技能匹配、资源独占）转化为实时掩码矩阵，确保输出动作物理可行 |

---

## 2. 空间站场景设定

### 2.1 舱段分区与物理约束

本系统将空间站作业空间划分为三个功能迥异的拓扑区域：

| 区域 | 坐标范围 | 功能 | 任务类型 | 截止时间 |
|------|---------|------|---------|---------|
| **Zone A** (物资仓储舱) | x ∈ [0.0, 0.3] | 高密度货架环境，通道狭窄 | Task 1 (单人简单任务) | 200s |
| **Zone B** (能源动力舱) | x ∈ [0.4, 0.7] | 高压设备区，强电磁干扰 | Task 2 (双人协同任务) | 60s |
| **Zone C** (核心控制舱) | x ∈ [0.8, 1.0] | 主控大厅，空间开阔 | Task 3 (三人协同任务) | 48s |

### 2.2 异构机器人定义

系统包含 **3 种固定角色**的异构机器人，每种具有唯一的能力向量和差异化运动速度：

| 类型 | 角色 | 能力向量 [视觉, 操作, 感知] | 速度 | 典型职责 |
|------|------|---------------------------|------|---------|
| Type 1 | Inspector（巡检机器人） | `[1, 0, 0]` | 0.5（快速） | 自由巡检、视觉检测 |
| Type 2 | Manipulator（操作机器人） | `[0, 1, 0]` | 0.1（慢速） | 精密操作、设备维修 |
| Type 3 | Sentinel（感知机器人） | `[0, 0, 1]` | 0.3（中等） | 环境感知、数据采集 |

### 2.3 任务分类

任务根据所需能力维度分为三类：

| 任务类型 | 需求向量 | 协作人数 | 时空特征 |
|---------|---------|---------|---------|
| Task 1 (简单) | `[1, 0, 0]` | 1 | 近端，宽松时限 |
| Task 2 (中等) | `[0, 1, 1]` | 2 | 中段，中等时限 |
| Task 3 (困难) | `[1, 1, 1]` | 3 | 远端，极限时限 |

---

## 3. 代码结构

```
├── attention.py          # 核心网络：多头注意力、编码器/解码器、指针网络、GateFFN
├── driver.py             # 训练主循环：REINFORCE 算法、Ray 分布式调度、模型保存/加载
├── runner.py             # Ray Actor 封装：负责权重同步与训练/测试任务分发
├── worker.py             # 单个 episode 执行：环境交互、经验收集、POMO 采样
├── task_env.py           # 环境核心：任务生成、机器人调度、约束裁剪、动画可视化
├── parameters.py         # 超参数配置：环境参数、训练参数、保存参数
├── test.py               # 测试脚本：加载模型、批量推理、CSV 结果导出
├── quick_view.py         # 快速可视化：Greedy 策略推演与 GIF 动图生成
├── yamlGenerator.py      # YAML 配置生成器：为 CTAS-D 对比方法生成参数文件
├── import torch.py       # GPU/CUDA 环境检测工具
└── README.md             # 本文件
```

### 3.1 核心模块说明

#### `attention.py` — 策略网络

- **`SingleHeadAttention`**: 指针注意力（Pointer Attention），输出任务选择的概率分布
- **`MultiHeadAttention`**: 多头注意力，支持编码器自注意力与解码器交叉注意力
- **`GateFFNDense`**: 门控前馈网络（GLU 变体），替代传统 ReLU FFN
- **`Encoder` / `Decoder`**: 标准 Transformer 编解码器堆叠
- **`AttentionNet`**: 主网络，包含：
  - 双流编码器（Agent Encoder + Task Encoder）
  - 双路交叉注意力（Agent→Task 能力评估 + Task→Agent 需求评估）
  - 特征融合层（拼接三路特征：Agent-Task 交叉特征 + 全局任务聚合 + 全局智能体聚合）
  - 全局解码器 + 指针网络 → 输出动作概率

#### `task_env.py` — 任务环境

- **`generate_task()`**: 按概率生成三类任务（20% Task1, 30% Task2, 50% Task3）
- **`generate_agent()`**: 强制定义三种固定角色的能力向量
- **任务时空分布**：任务 x 坐标按 Zone 分布，y 坐标随机，deadline 按任务类型设定
- **机器人速度差异**：Type1=0.5, Type2=0.1, Type3=0.3
- **超时失败机制**：超过 deadline 且未开工的任务标记 `failed=True`，并从后续决策中移除
- **奖励函数**：`reward = -makespan - efficiency×10 - fail_count×200`

#### `driver.py` — 训练流程

- **分布式训练**：基于 Ray 框架，多 Worker 并行采集经验
- **REINFORCE 算法**：策略梯度 + 基线（Baseline）方差缩减
- **POMO 采样**：每 episode 多次采样取最优，提升解的质量
- **动态难度提升**：随训练进程逐步增加任务/机器人规模
- **统计评估**：定期在测试集上评估，基于 t 检验判定是否更新基线模型

#### `parameters.py` — 关键超参数

```python
# 环境参数（空间站场景）
EnvParams.SPECIES_AGENTS_RANGE = (2, 4)   # 每种机器人的数量范围
EnvParams.SPECIES_RANGE = (3, 3)          # 固定 3 种角色
EnvParams.TASKS_RANGE = (20, 40)          # 任务数量范围
EnvParams.TRAIT_DIM = 3                   # 能力维度：视觉、操作、感知
EnvParams.MAX_TIME = 200                  # 最大仿真时间

# 网络与训练参数
TrainParams.EMBEDDING_DIM = 128           # 嵌入维度
TrainParams.AGENT_INPUT_DIM = 9           # 6 + TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 11           # 5 + 2 × TRAIT_DIM
TrainParams.BATCH_SIZE = 512
TrainParams.LR = 1e-5
TrainParams.POMO_SIZE = 10
```

---

## 4. 快速开始

### 4.1 环境依赖

```
Python >= 3.6
PyTorch >= 1.8.1
numpy
ray
matplotlib
scipy
pandas
```

### 4.2 训练

1. 修改 `parameters.py` 中的超参数（如需调整场景规模或训练配置）
2. 运行训练：

```bash
python driver.py
```

训练过程将自动：
- 启动 Ray 分布式框架，创建多个并行 Worker
- 随机生成空间站场景（三类任务 × 三种机器人）
- 通过 REINFORCE 算法优化策略网络
- 定期评估并保存最优模型至 `model/{FOLDER_NAME}/`
- 记录训练指标至 TensorBoard（`train/{FOLDER_NAME}/`）

查看训练曲线：

```bash
tensorboard --logdir train/
```

### 4.3 测试

1. 确保已训练好的模型位于 `model/` 目录下
2. 修改 `test.py` 中的路径配置
3. 运行测试：

```bash
python test.py
```

测试将加载测试集 `.pkl` 文件，批量推理并输出 CSV 结果至 `对比效果和测试csv文件位置/csv文件保存/` 目录。

### 4.4 快速可视化（Greedy 基线）

```bash
python quick_view.py
```

将生成 Greedy 策略的执行动画 GIF，用于直观对比 RL 策略的效果。

---

## 5. 关键设计决策

### 5.1 与原始论文的差异

| 维度 | 原始论文 (Dai et al.) | 本项目修改 |
|------|---------------------|-----------|
| 能力维度 | 5 维通用技能 | **3 维**（视觉、操作、感知），对应空间站场景 |
| 机器人种类 | 随机生成，种类可变 | **固定 3 种角色**，能力向量硬编码 |
| 任务生成 | 随机需求 | **三类分级**：20% 单人 / 30% 双人 / 50% 三人 |
| 空间分布 | 均匀随机 | **三区域分区**（Zone A/B/C），任务类型与区域绑定 |
| 截止时间 | 无 | **按任务类型设定 deadline**，超时即失败 |
| 机器人速度 | 统一 | **差异化速度**：Type1 快 / Type2 慢 / Type3 中等 |
| 奖励函数 | 基于 makespan | 增加**失败惩罚项**（-200/task），强制 RL 关注成功率 |
| 动画可视化 | 基础样式 | **增强可视化**：分区标注、任务形状/颜色编码、机器人轨迹 |

### 5.2 训练配置说明

- `SaverParams.FOLDER_NAME = '3_11_FGH_training'`：当前训练结果保存目录
- `SaverParams.LOAD_MODEL = True`：从已有 checkpoint 恢复训练
- `SaverParams.LOAD_FROM = 'best'`：加载历史最优模型权重
- `TrainParams.FORCE_MAX_OPEN_TASK = True`：强制启用最大开放任务约束，减少死锁

---

## 6. 评估指标

系统在训练和测试过程中记录以下性能指标：

| 指标 | 含义 |
|------|------|
| **Success Rate** | 任务完成率（扣除超时失败的任务） |
| **Makespan** | 系统完工时间（所有机器人返回 depot 的最晚时刻） |
| **Time Cost** | 任务平均启动时间 |
| **Waiting Time** | 机器人平均等待时间 |
| **Travel Distance** | 机器人总行驶距离 |
| **Efficiency** | 等待效率（衡量能力冗余度） |

---

## 7. 参考文献

[1] W. Dai, U. Rai, J. Chiun, Y. Cao, G. Sartoretti, "Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning," *IEEE Robotics and Automation Letters (RAL)*, 2024.

[2] B. Fu, W. Smith, D. M. Rizzo, M. Castanier, M. Ghaffari, K. Barton, "Robust task scheduling for heterogeneous robot teams under capability uncertainty," *IEEE Transactions on Robotics*, 2022.

[3] Y.-D. Kwon, J. Choo, B. Kim, et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement Learning," *Advances in Neural Information Processing Systems*, 2020.

[4] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention Is All You Need," *Advances in Neural Information Processing Systems*, 2017.

---

## 8. 许可证

本项目的原始开源代码遵循 MIT 许可证。本项目的修改部分仅用于学术研究目的。
