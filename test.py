import os
import torch
from attention import AttentionNet
from worker import Worker
import numpy as np
from env.task_env import TaskEnv
import time
import pickle
import pandas as pd
import glob
from natsort import natsorted
import multiprocessing
from parameters import EnvParams, TrainParams

EnvParams.TASKS_RANGE = (20, 20)
EnvParams.SPECIES_RANGE = (3, 3)
EnvParams.SPECIES_AGENTS_RANGE = (3, 3)
EnvParams.MAX_TIME = 200
EnvParams.TRAIT_DIM = 3
TrainParams.EMBEDDING_DIM = 128
TrainParams.AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM + 2

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0
NUM_META_AGENT = 1
GAMMA = 1
FOLDER_NAME = '3_25_HRL_training_2'        
testSet = '3_25_单层陷阱测试集'        # 读取测试集pkl文件的路径
model_path = 'model/3_25_HRL_training_2'   # 读取训练好的模型路径
sampling = False
max_task = False
sampling_num = 10 if sampling else 1
save_img = False

import os

def main(f):
    device = torch.device('cuda:0') if USE_GPU_GLOBAL else torch.device('cpu')
    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    checkpoint = torch.load(f'{model_path}/checkpoint.pth', map_location=torch.device('cpu'))
    global_network.load_state_dict(checkpoint['best_model'])
    worker = Worker(0, global_network, global_network, 0, device)
    index = int(os.path.splitext(os.path.basename(f))[0].replace('env_', ''))
    #index = int(f.split('/')[-1].replace('.pkl', '').replace('env_', ''))
    env = pickle.load(open(f, 'rb'))
    results_best = None
    start = time.time()
    for i in range(sampling_num):
        env.init_state()
        worker.env = env
        _, _, results, _ = worker.run_episode(False, sampling, max_task)
        # print(results)
        if results_best is None:
            results_best = results
        else:
            if results_best['makespan'] >= results['makespan']:
                results_best = results
    if save_img:
        env.plot_animation(f'{testSet}/env_{index}', index)
    end = time.time() - start
    df_ = pd.DataFrame(results_best, index=[index])
    print(f)
    return df_, end


import multiprocessing as mp

def run_all():
    files = natsorted(glob.glob(f'{testSet}/env*.pkl'), key=lambda y: y.lower())
    b = []

    with mp.Pool(processes=1) as pool:
        final_results = pool.map(main, files)

    perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    df = pd.DataFrame(perf_metrics)

    for r in final_results:
        df = pd.concat([df, r[0]])
        b.append(r[1])

    print(np.mean(b))

     # 1. 获取当前test.py文件所在的绝对目录（跨电脑/跨系统通用）
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 拼接相对路径：和原绝对路径的“对比效果和测试csv文件位置\csv文件保存”保持一致
    #    注意：相对路径的层级要和你项目实际目录结构匹配！
    base_absolute_dir = os.path.join(
        current_script_dir,
        '对比效果和测试csv文件位置',
        'csv文件保存'
    )
    
    # 2. 测试结果保存的文件夹，依然复用 FOLDER_NAME 作为子文件夹
    target_dir = os.path.join(base_absolute_dir, FOLDER_NAME)

    # 3. 确保文件夹存在（递归创建，不存在则新建）
    os.makedirs(target_dir, exist_ok=True)
    print(f"📂 确保文件夹存在: {target_dir}")

    # 4. 测试结果保存文件名，拼接CSV文件名
    custom_filename = "25陷阱测试集跑25双层算法2"  # 你自己的名字
    output_filename = os.path.join(target_dir, f'{custom_filename}.csv')

    # 5. 保存文件
    df.to_csv(output_filename, index=True)
    print(f"💾 结果已成功保存至: {output_filename}")


if __name__ == "__main__":
    mp.freeze_support()  # 可选，但在 Windows 上更稳
    run_all()
