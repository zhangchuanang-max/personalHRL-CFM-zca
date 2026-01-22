import os
import numpy as np
import time
from env.task_env import TaskEnv

# 1. 定义保存动图的文件夹路径
# 名字见名知意：用于存放调试场景的动图
output_dir = './debug_scenarios'

# 如果文件夹不存在，自动创建（避免报错）
os.makedirs(output_dir, exist_ok=True)

print(f"正在初始化环境... 动图将保存在: {output_dir}")

# 2. 初始化环境（强制使用空间站场景参数）
# 注意：species_range 必须锁死为 (3, 3) 以匹配您修改后的 generate_agent
env = TaskEnv(
    per_species_range=(2, 2),  # 每种机器人各2个，方便观察
    species_range=(3, 3),      # 【重要】必须固定为3种，对应 Type 1/2/3
    tasks_range=(15, 15),      # 任务数量适中
    traits_dim=3,              # 【重要】能力维度改为3
    seed=42,                   # 固定随机种子，保证每次运行的任务分布位置一样
    plot_figure=True           # 开启绘图开关
)

# 3. 重置环境状态
env.init_state()
# 1. 直接定义算法名称
algorithm_name = "Greedy" 

print(f"开始执行策略推演...")

# 2. 执行推演
# 注意：我们只需传入简单的算法名，具体的“时间”后缀已在 task_env.py 中自动生成
makespan = env.execute_greedy_action(
    path=output_dir, 
    method=algorithm_name, 
    plot_figure=True
)

print("-" * 30)
print(f"推演结束！完工时间: {makespan:.2f}")
print(f"动图已保存至: {output_dir}")