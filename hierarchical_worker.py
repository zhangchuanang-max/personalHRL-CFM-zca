import pickle
import time
import torch
import numpy as np
import random
from env.hierarchical_task_env import HierarchicalTaskEnv
# ======== 【修改 1：导入你刚才写好的分层网络】 ========
from hierarchical_attention import HierarchicalAttentionNet
import scipy.signal as signal
from parameters import *
import copy
from torch.nn import functional as F
from torch.distributions import Categorical


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def zero_padding(x, padding_size, length):
    pad = torch.nn.ZeroPad2d((0, 0, 0, padding_size - length))
    x = pad(x)
    return x

# ======== 【修改 2：类名改为 HierarchicalWorker】 ========
class HierarchicalWorker:
    def __init__(self, mete_agent_id, local_network, local_baseline, global_step,
                 device='cuda', save_image=False, seed=None, env_params=None):

        self.device = device
        self.metaAgentID = mete_agent_id
        self.global_step = global_step
        self.save_image = save_image
        if env_params is None:
            env_params = [EnvParams.SPECIES_AGENTS_RANGE, EnvParams.SPECIES_RANGE, EnvParams.TASKS_RANGE]
        self.env = HierarchicalTaskEnv(*env_params, EnvParams.TRAIT_DIM, EnvParams.DECISION_DIM, seed=seed, plot_figure=save_image)
        self.baseline_env = copy.deepcopy(self.env)
        self.local_baseline = local_baseline
        self.local_net = local_network
        self.experience = {idx:[] for idx in range(7)}
        self.episode_number = None
        self.perf_metrics = {}
        self.p_rnn_state = {}
        self.max_time = EnvParams.MAX_TIME

    def run_episode(self, training=True, sample=False, max_waiting=False):
        buffer_dict = {idx:[] for idx in range(7)}
        perf_metrics = {}
        current_action_index = 0
        decision_step = 0
        
        # ======== 【HRL 核心变量：记录当前的宏观阶段 (阶段级调度)】 ========
        # -1 代表当前没有分配阶段，需要呼叫上层大脑 (Stage Controller) 做决定
        current_macro_stage = -1 
        
        while not self.env.finished and self.env.current_time < EnvParams.MAX_TIME and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.env.next_decision()
                self.env.current_time = current_time
                random.shuffle(release_agents[0])
                finished_task = []
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.env.agent_dic[agent_id]
                    # 1. 物理观测 (获取底层的 C_skill 和 C_access 硬约束)
                    task_info, total_agents, physical_mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting))
                    
                    if training:
                        task_info_pad, total_agents_pad, physical_mask_pad = self.obs_padding(task_info, total_agents, physical_mask)
                    else:
                        task_info_pad, total_agents_pad, physical_mask_pad = task_info, total_agents, physical_mask

                    # 2. 宏观决策：如果当前没有阶段目标，让统筹器选一个舱段 (0:ZoneA, 1:ZoneB, 2:ZoneC)
                    if current_macro_stage == -1:
                        macro_probs, _ = self.local_net.get_macro_policy(task_info_pad, total_agents_pad)
                        if training:
                            macro_action = Categorical(macro_probs).sample()
                        else:
                            macro_action = torch.argmax(macro_probs, dim=1)
                        current_macro_stage = macro_action.item()

                    # 3. 动态动作空间裁剪 (HRL核心！基于宏观阶段生成局部活跃子图)
                    sub_task_mask = physical_mask_pad.clone()
                    for t_idx in range(1, self.env.tasks_num + 1):
                        task_node = self.env.task_dic[t_idx - 1]
                        x_loc = task_node['location'][0]
                        # 将不在当前宏观区域内的任务强制置为 True (屏蔽)
                        if current_macro_stage == 0 and x_loc >= 0.4:
                            sub_task_mask[0, t_idx] = True 
                        elif current_macro_stage == 1 and (x_loc < 0.4 or x_loc >= 0.8):
                            sub_task_mask[0, t_idx] = True 
                        elif current_macro_stage == 2 and x_loc < 0.8:
                            sub_task_mask[0, t_idx] = True 

                    # 4. 检查拦截：如果在当前物理+宏观双重约束下，该机器人没有合法动作，则直接待机
                    block_flag = sub_task_mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                        
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    
                    # 3. 微观决策：调用下层网络，在 sub_task_mask 约束下安全选任务
                    probs, _ = self.local_net.get_micro_policy(task_info, total_agents, sub_task_mask, index)
                    
                    if training:
                        action = Categorical(probs).sample()
                        while action.item() > self.env.tasks_num:
                            action = Categorical(probs).sample()
                    else:
                        if sample:
                            action = Categorical(probs).sample()
                        else:
                            action = torch.argmax(probs, dim=1)
                            
                    r, doable, f_t = self.env.agent_step(agent_id, action.item(), decision_step)
                    agent['current_action_index'] = current_action_index
                    finished_task.append(f_t)
                    
                    if training and doable:
                        buffer_dict[0] += total_agents
                        buffer_dict[1] += task_info
                        buffer_dict[2] += action.unsqueeze(0)
                        buffer_dict[3] += mask # 此处仍保存原始mask用于loss计算
                        buffer_dict[4] += torch.FloatTensor([[0]]).to(self.device)  # reward
                        buffer_dict[5] += index
                        buffer_dict[6] += torch.FloatTensor([[0]]).to(self.device)
                        current_action_index += 1
                        
                    # ==============================================================
                    
                self.env.finished = self.env.check_finished()
                
                # 如果这个阶段所有子任务都被掩码了（做完了），重置宏观阶段
                # 迫使下一步进行新的宏观决策，这体现了 SMDP 的事件驱动特性
                # 此处为简化逻辑，后续需结合环境状态完善
                current_macro_stage = -1 
                
                decision_step += 1

        terminal_reward, finished_tasks = self.env.get_episode_reward(self.max_time)

        perf_metrics['success_rate'] = [np.sum(finished_tasks)/len(finished_tasks)]
        perf_metrics['makespan'] = [self.env.current_time]
        perf_metrics['time_cost'] = [np.nanmean(self.env.get_matrix(self.env.task_dic, 'time_start'))]
        perf_metrics['waiting_time'] = [np.mean(self.env.get_matrix(self.env.agent_dic, 'sum_waiting_time'))]
        perf_metrics['travel_dist'] = [np.sum(self.env.get_matrix(self.env.agent_dic, 'travel_dist'))]
        perf_metrics['efficiency'] = [self.env.get_efficiency()]
        return terminal_reward, buffer_dict, perf_metrics

    def baseline_test(self):
        self.baseline_env.plot_figure = False
        perf_metrics = {}
        current_action_index = 0
        start = time.time()
        
        current_macro_stage = -1 # 测试时同样需要分层逻辑
        
        while not self.baseline_env.finished and self.baseline_env.current_time < self.max_time and current_action_index < 300:
            with torch.no_grad():
                release_agents, current_time = self.baseline_env.next_decision()
                random.shuffle(release_agents[0])
                self.baseline_env.current_time = current_time
                if time.time() - start > 30:
                    break
                while release_agents[0] or release_agents[1]:
                    agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                    agent = self.baseline_env.agent_dic[agent_id]
                    task_info, total_agents, mask = self.convert_torch(self.baseline_env.agent_observe(agent_id, False))
                    return_flag = mask[0, 1:].all().item()
                    if return_flag and not np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')): 
                        self.baseline_env.agent_dic[agent_id]['no_choice'] = return_flag
                        continue
                    elif return_flag and np.all(self.baseline_env.get_matrix(self.baseline_env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                        
                    # 1. 物理观测 (获取底层的 C_skill 和 C_access 硬约束)
                    task_info, total_agents, physical_mask = self.convert_torch(self.env.agent_observe(agent_id, max_waiting))
                    
                    if training:
                        task_info_pad, total_agents_pad, physical_mask_pad = self.obs_padding(task_info, total_agents, physical_mask)
                    else:
                        task_info_pad, total_agents_pad, physical_mask_pad = task_info, total_agents, physical_mask

                    # 2. 宏观决策：如果当前没有阶段目标，让统筹器选一个舱段 (0:ZoneA, 1:ZoneB, 2:ZoneC)
                    if current_macro_stage == -1:
                        macro_probs, _ = self.local_net.get_macro_policy(task_info_pad, total_agents_pad)
                        if training:
                            macro_action = Categorical(macro_probs).sample()
                        else:
                            macro_action = torch.argmax(macro_probs, dim=1)
                        current_macro_stage = macro_action.item()

                    # 3. 动态动作空间裁剪 (HRL核心！基于宏观阶段生成局部活跃子图)
                    sub_task_mask = physical_mask_pad.clone()
                    for t_idx in range(1, self.env.tasks_num + 1):
                        task_node = self.env.task_dic[t_idx - 1]
                        x_loc = task_node['location'][0]
                        # 将不在当前宏观区域内的任务强制置为 True (屏蔽)
                        if current_macro_stage == 0 and x_loc >= 0.4:
                            sub_task_mask[0, t_idx] = True 
                        elif current_macro_stage == 1 and (x_loc < 0.4 or x_loc >= 0.8):
                            sub_task_mask[0, t_idx] = True 
                        elif current_macro_stage == 2 and x_loc < 0.8:
                            sub_task_mask[0, t_idx] = True 

                    # 4. 检查拦截：如果在当前物理+宏观双重约束下，该机器人没有合法动作，则直接待机
                    block_flag = sub_task_mask[0, 1:].all().item()
                    if block_flag and not np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')):
                        agent['no_choice'] = block_flag
                        continue
                    elif block_flag and np.all(self.env.get_matrix(self.env.task_dic, 'feasible_assignment')) and agent['current_task'] < 0:
                        continue
                        
                    index = torch.LongTensor([agent_id]).reshape(1, 1, 1).to(self.device)
                    
                    probs, _ = self.local_baseline.get_micro_policy(task_info, total_agents, sub_task_mask, index)
                    action = torch.argmax(probs, 1)
                    
                    self.baseline_env.agent_step(agent_id, action.item(), None)
                    current_action_index += 1
                    
                self.baseline_env.finished = self.baseline_env.check_finished()
                current_macro_stage = -1 

        reward, finished_tasks = self.baseline_env.get_episode_reward(self.max_time)
        return reward

    def work(self, episode_number):
        baseline_rewards = []
        buffers = []
        metrics = []
        max_waiting = TrainParams.FORCE_MAX_OPEN_TASK
        for _ in range(TrainParams.POMO_SIZE):
            self.env.init_state()
            terminal_reward, buffer, perf_metrics = self.run_episode(episode_number,True, max_waiting)
            if terminal_reward is np.nan:
                max_waiting = True
                continue
            baseline_rewards.append(terminal_reward)
            buffers.append(buffer)
            metrics.append(perf_metrics)
        baseline_reward = np.nanmean(baseline_rewards)

        for idx, buffer in enumerate(buffers):
            for key in buffer.keys():
                if key == 6:
                    for i in range(len(buffer[key])):
                        buffer[key][i] += baseline_rewards[idx] - baseline_reward
                if key not in self.experience.keys():
                    self.experience[key] = buffer[key]
                else:
                    self.experience[key] += buffer[key]

        for metric in metrics:
            for key in metric.keys():
                if key not in self.perf_metrics.keys():
                    self.perf_metrics[key] = metric[key]
                else:
                    self.perf_metrics[key] += metric[key]

        if self.save_image:
            try:
                self.env.plot_animation(SaverParams.GIFS_PATH, episode_number)
            except:
                pass
        self.episode_number = episode_number

    def convert_torch(self, args):
        data = []
        for arg in args:
            data.append(torch.tensor(arg, dtype=torch.float).to(self.device))
        return data

    @staticmethod
    def obs_padding(task_info, agents, mask):
        task_info = F.pad(task_info, (0, 0, 0, EnvParams.TASKS_RANGE[1] + 1 - task_info.shape[1]), 'constant', 0)
        agents = F.pad(agents, (0, 0, 0, EnvParams.SPECIES_AGENTS_RANGE[1] * EnvParams.SPECIES_RANGE[1] - agents.shape[1]), 'constant', 0)
        mask = F.pad(mask, (0, EnvParams.TASKS_RANGE[1] + 1 - mask.shape[1]), 'constant', 1)
        return task_info, agents, mask


if __name__ == '__main__':
    device = torch.device('cuda')
    # ==== 测试时如果直接运行本文件，也要导入分层网络 ====
    localNetwork = HierarchicalAttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM).to(device)
    for i in range(100):
        worker = HierarchicalWorker(1, localNetwork, localNetwork, 0, device=device, seed=i, save_image=False)
        worker.work(i)
        print(i)