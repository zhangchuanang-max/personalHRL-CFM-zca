import os
os.environ['OMP_NUM_THREADS'] = '1'  
os.environ['MKL_NUM_THREADS'] = '1'

import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import numpy as np
import random

from attention import AttentionNet
from runner import RLRunner
from parameters import *
from env.task_env import TaskEnv
from scipy.stats import ttest_rel
from torch.distributions import Categorical


class Logger(object):
    def __init__(self):
        self.global_net = None
        self.baseline_net = None
        self.optimizer = None
        self.lr_decay = None
        self.writer = SummaryWriter(SaverParams.TRAIN_PATH)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.MODEL_PATH, exist_ok=True)
        if SaverParams.SAVE:
            os.makedirs(SaverParams.GIFS_PATH, exist_ok=True)

    def set(self,  global_net, baseline_net, optimizer, lr_decay):
        self.global_net = global_net
        self.baseline_net = baseline_net
        self.optimizer = optimizer
        self.lr_decay = lr_decay

    def write_to_board(self, tensorboard_data, curr_episode):
        tensorboard_data = np.array(tensorboard_data)
        tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
        reward, p_l, entropy, grad_norm, z_loss, t_loss, success_rate, time, time_cost, waiting, distance, effi = tensorboard_data
        print(f"🚀 [进度汇报] Episode: {curr_episode} | 成功率: {success_rate:.1%} | 奖励Reward: {reward:.0f} | 总Loss: {t_loss:.4f} | Zone Loss: {z_loss:.4f}")
        metrics = {'Loss/Learning Rate': self.lr_decay.get_last_lr()[0],
                   'Loss/Policy Loss': p_l,
                   'Loss/Entropy': entropy,
                   'Loss/Grad Norm': grad_norm,
                   'Loss/zone_loss': z_loss,          
                   'Loss/total_loss': t_loss,         
                   'Loss/Reward': reward,
                   'Perf/Makespan': time,
                   'Perf/Success rate': success_rate,
                   'Perf/Time cost': time_cost,
                   'Perf/Waiting time': waiting,
                   'Perf/Traveling distance':distance,
                   'Perf/Waiting Efficiency': effi
                   }
        for k, v in metrics.items():
            self.writer.add_scalar(tag=k, scalar_value=v, global_step=curr_episode)

    def load_saved_model(self):
        print('Loading Model...')
        checkpoint = torch.load(SaverParams.MODEL_PATH + '/checkpoint.pth')
        if SaverParams.LOAD_FROM == 'best':
            model = 'best_model'
        else:
            model = 'model'
        self.global_net.load_state_dict(checkpoint[model])
        self.baseline_net.load_state_dict(checkpoint[model])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        curr_level = checkpoint['level']
        best_perf = checkpoint['best_perf']
        print("curr_episode set to ", curr_episode)
        print("best_perf so far is ", best_perf)
        print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        if TrainParams.RESET_OPT:
            self.optimizer = optim.Adam(self.global_net.parameters(), lr=TrainParams.LR)
            self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)
        return curr_episode, curr_level, best_perf

    def save_model(self, curr_episode, curr_level, best_perf):
        print('Saving model', end='\n')
        checkpoint = {"model": self.global_net.state_dict(),
                      "best_model": self.baseline_net.state_dict(),
                      "best_optimizer": self.optimizer.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "episode": curr_episode,
                      "lr_decay": self.lr_decay.state_dict(),
                      "level": curr_level,
                      "best_perf": best_perf
                      }
        path_checkpoint = "./" + SaverParams.MODEL_PATH + "/checkpoint.pth"
        torch.save(checkpoint, path_checkpoint)
        print('Saved model', end='\n')

    @staticmethod
    def generate_env_params(curr_level=None):
        per_species_num = np.random.randint(EnvParams.SPECIES_AGENTS_RANGE[0], EnvParams.SPECIES_AGENTS_RANGE[1] + 1)
        species_num = np.random.randint(EnvParams.SPECIES_RANGE[0], EnvParams.SPECIES_RANGE[1] + 1)
        tasks_num = np.random.randint(EnvParams.TASKS_RANGE[0], EnvParams.TASKS_RANGE[1] + 1)
        params = [(per_species_num, per_species_num), (species_num, species_num), (tasks_num, tasks_num)]
        return params

    @staticmethod
    def generate_test_set_seed():
        test_seed = np.random.randint(low=0, high=1e8, size=TrainParams.EVALUATION_SAMPLES).tolist()
        return test_seed


def fuse_two_dicts(ini_dictionary1, ini_dictionary2):
    if ini_dictionary2 is not None:
        merged_dict = {**ini_dictionary1, **ini_dictionary2}
        final_dict = {}
        for k, v in merged_dict.items():
            final_dict[k] = ini_dictionary1[k] + v
        return final_dict
    else:
        return ini_dictionary1


def main():
    logger = Logger()
    ray.init()
    device = torch.device('cuda') if TrainParams.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')

    global_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    baseline_network = AttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    global_optimizer = optim.Adam(global_network.parameters(), lr=TrainParams.LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=TrainParams.DECAY_STEP, gamma=0.98)

    logger.set(global_network, baseline_network, global_optimizer, lr_decay)

    curr_episode = 0
    curr_level = 0
    best_perf = -200
    if SaverParams.LOAD_MODEL:
        curr_episode, curr_level, best_perf = logger.load_saved_model()
        if curr_level is None:
            curr_level = 0
            
        # 如果恢复模型时已经度过了冷启动阶段，直接解冻Manager
        if curr_episode >= TrainParams.ENABLE_MANAGER_AFTER:
            global_network._force_uniform_manager = False

    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        baseline_weights = baseline_network.to(local_device).state_dict()
        global_network.to(device)
        baseline_network.to(device)
    else:
        weights = global_network.state_dict()
        baseline_weights = baseline_network.state_dict()
    weights_memory = ray.put(weights)
    baseline_weights_memory = ray.put(baseline_weights)

    jobs = []
    env_params = logger.generate_env_params(curr_level)
    for i, meta_agent in enumerate(meta_agents):
        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
        curr_episode += 1
    test_set = logger.generate_test_set_seed()
    baseline_value = None
    
    experience_buffer = {idx:[] for idx in range(9)}
    perf_metrics = {'success_rate': [], 'makespan': [], 'time_cost': [], 'waiting_time': [], 'travel_dist': [], 'efficiency': []}
    training_data = []

    try:
        while True:
            # 【新增】解冻Manager的控制台提醒
            if curr_episode == TrainParams.ENABLE_MANAGER_AFTER:
                print(f"\n=============================================\n🔓 [Manager解冻] Episode {curr_episode}: 开始正式训练Manager!\n=============================================\n")
                global_network._force_uniform_manager = False

            done_id, jobs = ray.wait(jobs)
            done_job = ray.get(done_id)[0]
            buffer, metrics, info = done_job
            experience_buffer = fuse_two_dicts(experience_buffer, buffer)
            perf_metrics = fuse_two_dicts(perf_metrics, metrics)

            update_done = False
            if len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                train_metrics = []
                while len(experience_buffer[0]) >= TrainParams.BATCH_SIZE:
                    rollouts = {}
                    for k, v in experience_buffer.items():
                        rollouts[k] = v[:TrainParams.BATCH_SIZE]
                    for k in experience_buffer.keys():
                        experience_buffer[k] = experience_buffer[k][TrainParams.BATCH_SIZE:]
                    if len(experience_buffer[0]) < TrainParams.BATCH_SIZE:
                        update_done = True
                    if update_done:
                        for v in experience_buffer.values():
                            del v[:]

                    agent_inputs = torch.stack(rollouts[0], dim=0).to(device)  
                    task_inputs = torch.stack(rollouts[1], dim=0).to(device)  
                    action_batch = torch.stack(rollouts[2], dim=0).unsqueeze(1).to(device)  
                    global_mask_batch = torch.stack(rollouts[3], dim=0).to(device)  
                    reward_batch = torch.stack(rollouts[4], dim=0).unsqueeze(1).to(device)  
                    index = torch.stack(rollouts[5]).to(device)
                    advantage_batch = torch.stack(rollouts[6], dim=0).to(device)  
                    zone_choice_list = rollouts[7]
                    zone_choice_batch = torch.cat(zone_choice_list, dim=0).to(device) if len(zone_choice_list) > 0 else None
                    manager_adv_batch = torch.stack(rollouts[8], dim=0).to(device)

                    probs, zone_probs_batch, _ = global_network(task_inputs, agent_inputs, global_mask_batch, index, fixed_zone_choice=zone_choice_batch)
                    
                    dist = Categorical(probs)
                    logp = dist.log_prob(action_batch.flatten())
                    entropy = dist.entropy().mean()
                    
                    # 1. Worker Loss
                    worker_advantages = advantage_batch.flatten().detach()
                    worker_loss = - (logp * worker_advantages).mean()

                    # 2. Manager Loss
                    if zone_probs_batch is not None and zone_choice_batch is not None:
                        chosen_zone_prob = torch.sum(zone_probs_batch * zone_choice_batch, dim=-1)
                        zone_log_probs = torch.log(chosen_zone_prob + 1e-8)
                        
                        manager_advantages = manager_adv_batch.flatten().detach()
                        manager_loss = -torch.mean(manager_advantages * zone_log_probs)
                        
                        zone_entropy = -torch.sum(zone_probs_batch * torch.log(zone_probs_batch + 1e-8), dim=-1)
                        zone_entropy_bonus = 0.05 * zone_entropy.mean()
                        manager_loss = manager_loss - zone_entropy_bonus
                    else:
                        manager_loss = torch.tensor(0.0, device=worker_loss.device)

                    # 3. 合并总损失（渐进式训练权重控制）
                    if curr_episode < TrainParams.ENABLE_MANAGER_AFTER:
                        # 阶段1：只训练Worker，Manager输出均匀分布，不参与Loss
                        manager_loss = torch.tensor(0.0, device=worker_loss.device)
                        total_loss = worker_loss
                    elif curr_episode < TrainParams.ENABLE_MANAGER_AFTER + TrainParams.MANAGER_WARMUP_EPISODES:
                        # 阶段2：Manager预热期，权重从0.1线性增加到0.5
                        warmup_progress = (curr_episode - TrainParams.ENABLE_MANAGER_AFTER) / TrainParams.MANAGER_WARMUP_EPISODES
                        manager_weight = 0.05 + 0.25 * warmup_progress  
                        total_loss = worker_loss + manager_weight * manager_loss
                    else:
                        # 阶段3：正常双层训练
                        total_loss = worker_loss + 0.3 * manager_loss

                    global_optimizer.zero_grad()
                    total_loss.backward()
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                    global_optimizer.step()

                    train_metrics.append([reward_batch.mean().item(), worker_loss.item(), entropy.item(), grad_norm.item(), manager_loss.item(), total_loss.item()])
                lr_decay.step()

                perf_data = []
                for k, v in perf_metrics.items():
                    perf_data.append(np.nanmean(perf_metrics[k]))
                    del v[:]
                train_metrics = np.nanmean(train_metrics, axis=0)
                for v in perf_metrics.values():
                    del v[:]
                data = [*train_metrics, *perf_data]
                training_data.append(data)

            if len(training_data) >= TrainParams.SUMMARY_WINDOW:
                logger.write_to_board(training_data, curr_episode)
                training_data = []

            if update_done:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    baseline_weights = baseline_network.to(local_device).state_dict()
                    global_network.to(device)
                    baseline_network.to(device)
                else:
                    weights = global_network.state_dict()
                    baseline_weights = baseline_network.state_dict()
                weights_memory = ray.put(weights)
                baseline_weights_memory = ray.put(baseline_weights)

            env_params = logger.generate_env_params(curr_level)
            jobs.append(meta_agents[info['id']].training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
            curr_episode += 1

            if curr_episode // (TrainParams.INCREASE_DIFFICULTY * (curr_level + 1)) == 1 and curr_level < 10:
                curr_level += 1
                print('increase difficulty to level', curr_level)

            if curr_episode % 512 == 0:
                logger.save_model(curr_episode, curr_level, best_perf)

            if TrainParams.EVALUATE:
                if curr_episode % 1024 == 0:
                    ray.wait(jobs, num_returns=TrainParams.NUM_META_AGENT)
                    for a in meta_agents:
                        ray.kill(a)
                    print('Evaluate baseline model at ', curr_episode)

                    if baseline_value is None:
                        test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                        for _, test_agent in enumerate(test_agent_list):
                            ray.get(test_agent.set_baseline_weights.remote(baseline_weights_memory))
                        rewards = dict()
                        seed_list = copy.deepcopy(test_set)
                        evaluate_jobs = [test_agent_list[i].testing.remote(seed=seed_list.pop()) for i in range(TrainParams.NUM_META_AGENT)]
                        while True:
                            test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                            test_result = ray.get(test_done_id)[0]
                            reward, seed, meta_id = test_result
                            rewards[seed] = reward
                            if seed_list:
                                evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop()))
                            if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                                break
                        rewards = dict(sorted(rewards.items()))
                        baseline_value = np.stack(list(rewards.values()))
                        for a in test_agent_list:
                            ray.kill(a)

                    test_agent_list = [RLRunner.remote(metaAgentID=i) for i in range(TrainParams.NUM_META_AGENT)]
                    for _, test_agent in enumerate(test_agent_list):
                        ray.get(test_agent.set_baseline_weights.remote(weights_memory))
                    rewards = dict()
                    seed_list = copy.deepcopy(test_set)
                    evaluate_jobs = [test_agent_list[i].testing.remote(seed=seed_list.pop()) for i in range(TrainParams.NUM_META_AGENT)]
                    while True:
                        test_done_id, evaluate_jobs = ray.wait(evaluate_jobs)
                        test_result = ray.get(test_done_id)[0]
                        reward, seed, meta_id = test_result
                        rewards[seed] = reward
                        if seed_list:
                            evaluate_jobs.append(test_agent_list[meta_id].testing.remote(seed=seed_list.pop()))
                        if len(rewards) == TrainParams.EVALUATION_SAMPLES:
                            break
                    rewards = dict(sorted(rewards.items()))
                    test_value = np.stack(list(rewards.values()))
                    for a in test_agent_list:
                        ray.kill(a)

                    meta_agents = [RLRunner.remote(i) for i in range(TrainParams.NUM_META_AGENT)]

                    print('test value', test_value.mean())
                    print('baseline value', baseline_value.mean())
                    if test_value.mean() > baseline_value.mean():
                        _, p = ttest_rel(test_value, baseline_value)
                        print('p value', p)
                        if p < 0.05:
                            print('update baseline model at ', curr_episode)
                            if device != local_device:
                                weights = global_network.to(local_device).state_dict()
                                global_network.to(device)
                            else:
                                weights = global_network.state_dict()
                            baseline_weights = copy.deepcopy(weights)
                            baseline_network.load_state_dict(baseline_weights)
                            weights_memory = ray.put(weights)
                            baseline_weights_memory = ray.put(baseline_weights)
                            test_set = logger.generate_test_set_seed()
                            print('update test set')
                            baseline_value = None
                            best_perf = test_value.mean()
                            logger.save_model(curr_episode, curr_level, best_perf)
                    jobs = []
                    for i, meta_agent in enumerate(meta_agents):
                        jobs.append(meta_agent.training.remote(weights_memory, baseline_weights_memory, curr_episode, env_params))
                        curr_episode += 1

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

if __name__ == "__main__":
    main()