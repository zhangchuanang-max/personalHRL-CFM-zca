import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
# 1. 请输入您的两个 CSV 文件路径 (请根据实际情况修改文件名)
# 注意：确保这两个文件都在当前目录下，或者写绝对路径
file_rl = 'HeteroMRTA-main\SpaceStationTestSet\RL_sampling_False_1.csv'  # RL 的结果文件

file_greedy = 'HeteroMRTA-main\SpaceStationTestSet/greedy10.csv'   # Greedy 的结果文件 (假设您跑Greedy时保存了这个名字)

# 如果找不到 Greedy 的 csv，您可以手动创建一个临时的来测试脚本
# file_greedy = 'SpaceStationTestSet/Greedy_temp.csv' 
# ===========================================

def plot_comparison():
    # 1. 读取数据
    try:
        df_rl = pd.read_csv(file_rl)
        # 如果您的 Greedy 代码没有生成 CSV，只输出了数字，您可以手动创建一个只有一列 'makespan' 的 CSV
        df_greedy = pd.read_csv(file_greedy) 
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件。请检查文件名是否正确。\n详细信息: {e}")
        return

    # 提取 Makespan (完工时间) 列
    # 假设 CSV 里有一列叫 'makespan' (如果叫别的，比如 'cost'，请在这里修改)
    col_name = 'makespan' 
    
    if col_name not in df_rl.columns or col_name not in df_greedy.columns:
        print(f"❌ 列名错误：CSV 文件中必须包含 '{col_name}' 列。")
        print(f"RL 列名: {df_rl.columns}")
        print(f"Greedy 列名: {df_greedy.columns}")
        return

    rl_data = df_rl[col_name]
    greedy_data = df_greedy[col_name]

    # 2. 计算统计数据
    rl_mean = rl_data.mean()
    greedy_mean = greedy_data.mean()
    improvement = ((greedy_mean - rl_mean) / greedy_mean) * 100

    print("-" * 30)
    print(f"📊 实验结果对比摘要")
    print("-" * 30)
    print(f"Greedy 平均耗时: {greedy_mean:.2f}")
    print(f"RL (Ours) 平均耗时: {rl_mean:.2f}")
    print(f"🚀 效率提升 (Improvement): {improvement:.2f}%")
    print("-" * 30)

    # 3. 绘制图表
    plt.style.use('seaborn-v0_8-whitegrid') # 使用学术风格背景
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 左图：柱状图 (平均值对比) ---
    labels = ['Greedy (Baseline)', 'RL (Ours)']
    means = [greedy_mean, rl_mean]
    colors = ['#A9A9A9', '#1f77b4'] # 灰色代表基准，蓝色代表我们

    bars = ax1.bar(labels, means, color=colors, width=0.5, edgecolor='black')
    ax1.set_ylabel('Average Makespan (Time Steps)', fontsize=12)
    ax1.set_title('Efficiency Comparison (Lower is Better)', fontsize=14)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # --- 右图：箱线图 (分布稳定性对比) ---
    # 箱线图能看出是否有很多离群点（死锁导致的高耗时）
    box_data = [greedy_data, rl_data]
    
    bplot = ax2.boxplot(box_data, patch_artist=True, labels=labels, widths=0.5)
    
    # 上色
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    ax2.set_ylabel('Makespan Distribution', fontsize=12)
    ax2.set_title('Robustness Analysis', fontsize=14)

    # 保存图片
    save_name = 'Innovation1_Result_Comparison.png'
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"✅ 对比图已保存为: {save_name}")
    plt.show()

if __name__ == '__main__':
    plot_comparison()