import torch
import numpy as np
import ray
import os
# ======== 【修改 1：导入新的分层模块】 ========
from hierarchical_attention import HierarchicalAttentionNet
# 提前导入我们下一步将要新建的 HierarchicalWorker，防止覆盖原有逻辑
from hierarchical_worker import HierarchicalWorker
from parameters import *
from env.hierarchical_task_env import HierarchicalTaskEnv



class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if TrainParams.USE_GPU else torch.device('cpu')
        
        # ======== 【修改 2：实例化双层网络】 ========
        # 注意这里改成了 HierarchicalAttentionNet
        self.localNetwork = HierarchicalAttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM)
        self.localNetwork.to(self.device)
        self.localBaseline = HierarchicalAttentionNet(TrainParams.AGENT_INPUT_DIM, TrainParams.TASK_INPUT_DIM, TrainParams.EMBEDDING_DIM)
        self.localBaseline.to(self.device)

    def get_weights(self):
        return self.localNetwork.state_dict()

    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    def set_baseline_weights(self, weights):
        self.localBaseline.load_state_dict(weights)

    def training(self, global_weights, baseline_weights, curr_episode, env_params):
        print("starting episode {} on metaAgent {}".format(curr_episode, self.metaAgentID))
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)
        self.set_baseline_weights(baseline_weights)
        save_img = False
        if SaverParams.SAVE_IMG:
            if curr_episode % SaverParams.SAVE_IMG_GAP == 0:
                save_img = True
        # ======== 【修改 3：使用新的分层 Worker 跑训练】 ========
        worker = HierarchicalWorker(self.metaAgentID, self.localNetwork, self.localBaseline,
                        curr_episode, self.device, save_img, None, env_params)
        worker.work(curr_episode)

        buffer = worker.experience
        perf_metrics = worker.perf_metrics

        info = {
            "id": self.metaAgentID,
            "episode_number": curr_episode,
        }

        return buffer, perf_metrics, info

    def testing(self, seed=None):
        # ======== 【修改 4：使用新的分层 Worker 跑测试】 ========
        worker = HierarchicalWorker(self.metaAgentID, self.localNetwork, self.localBaseline,
                        0, self.device, False, seed)
        reward = worker.baseline_test()
        return reward, seed, self.metaAgentID


@ray.remote(num_cpus=1, num_gpus=TrainParams.NUM_GPU / TrainParams.NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):
        super().__init__(metaAgentID)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
