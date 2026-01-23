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
TrainParams.TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM

USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0
NUM_META_AGENT = 1
GAMMA = 1
FOLDER_NAME = 'save'
testSet = 'SpaceStationTestSet'
model_path = 'model/save_with_cfm'
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
        _, _, results = worker.run_episode(False, sampling, max_task)
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
    #df.to_csv(f'{testSet}/RL_sampling_{sampling}_{sampling_num}.csv', index=True)   #保存的csv文件名字，记得每次跑要修改
# ==========================================
    # 【核心修改】指定绝对路径保存结果
    # ==========================================
    # 1. 定义目标文件夹 (前面的 r 表示 raw string，解决 Windows 反斜杠转义问题)
    target_dir = r'D:\集群协同\二辩开题\HeteroMRTA（ours）\HeteroMRTA-main\SpaceStationTestSet'
    
    # 2. 自动拼接完整路径
    # 使用 os.path.join 可以自动处理路径连接符，更安全
    output_filename = os.path.join(target_dir, 'result_save_with_cfm.csv')
    
    # 3. 确保文件夹存在 (防止报错)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📂 已自动创建文件夹: {target_dir}")

    # 4. 保存文件
    df.to_csv(output_filename, index=True)
    print(f"💾 结果已成功保存至: {output_filename}")


if __name__ == "__main__":
    mp.freeze_support()  # 可选，但在 Windows 上更稳
    run_all()
