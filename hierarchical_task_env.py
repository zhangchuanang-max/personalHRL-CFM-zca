import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from itertools import combinations, product
import copy
from matplotlib.lines import Line2D  # <--- 【新增】用于创建自定义图例
# ...


class HierarchicalTaskEnv:
    def __init__(self, per_species_range=(10, 10), species_range=(5, 5), tasks_range=(30, 30), traits_dim=5,
                 decision_dim=10, max_task_size=2, duration_scale=5, seed=None, plot_figure=False):
        """
        :param traits_dim: number of capabilities in this problem, e.g. 3 traits
        :param seed: seed to generate pseudo random problem instance
        """
        self.rng = None
        self.per_species_range = per_species_range
        self.species_range = species_range
        self.tasks_range = tasks_range
        self.max_task_size = max_task_size
        self.duration_scale = duration_scale
        self.plot_figure = plot_figure
        # if seed is not None:
        #     self.rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed)
        self.traits_dim = traits_dim
        self.traits_dim = traits_dim
        self.decision_dim = decision_dim

        self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.species_distance_matrix, self.species_neighbor_matrix = self.generate_distance_matrix()
        # self.species_mask = self.calculate_optimized_ability()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        # self.best_route = self.calculate_tsp_route()

        self.current_time = 0
        self.dt = 0.1
        self.max_waiting_time = 200
        self.depot_waiting_time = 0
        self.finished = False
        self.reactive_planning = False

    def random_int(self, low, high, size=None):
        if self.rng is not None:
            integer = self.rng.integers(low, high, size)
        else:
            integer = np.random.randint(low, high, size)
        return integer

    def random_value(self, row, col):
        if self.rng is not None:
            value = self.rng.random((row, col))
        else:
            value = np.random.rand(row, col)
        return value

    def random_choice(self, a, size=None, replace=True):
        if self.rng is not None:
            choice = self.rng.choice(a, size, replace)
        else:
            choice = np.random.choice(a, size, replace)
        return choice
    
    def generate_task(self, tasks_num):
        tasks_ini = np.zeros((tasks_num, self.traits_dim))
        self.task_types = []
        
        for i in range(tasks_num):
            rand = self.rng.random() if self.rng else np.random.rand()
            
            # --- 修改开始 ---
            if rand < 0.2: 
                # 20% 生成 Task 1 (简单，单人)
                tasks_ini[i] = [1, 0, 0] 
                self.task_types.append(1)
                
            elif rand < 0.5: 
                # 30% 生成 Task 2 (中等，双人) -> 新增！
                # 假设 Task 2 需要 [0, 1, 1] 或者其他双技能组合，这里为了简单起见
                # 设定为需要第2和第3个技能 (模拟操作+感知)
                tasks_ini[i] = [0, 1, 1] 
                self.task_types.append(2)
                
            else:
                # 50% 生成 Task 3 (困难，三人)
                # 需要所有技能 [1, 1, 1]
                tasks_ini[i] = [1, 1, 1] 
                self.task_types.append(3)
            # --- 修改结束 ---
                
        return tasks_ini

    



    # def generate_agent(self, species_num):
    #     # agents_ini = self.random_value(species_num, self.traits_dim) > 0.8
    #     # while not np.all(np.sum(agents_ini, axis=1) != 0):
    #     #     agents_ini = self.random_value(species_num, self.traits_dim) > 0.8

    #     agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))
    #     while not np.all(np.sum(agents_ini, axis=1) != 0) or np.unique(agents_ini, axis=0).shape[0] != species_num:
    #         agents_ini = self.random_int(0, 2, (species_num, self.traits_dim))

    #     # agents_ini = np.diag(np.ones(self.traits_dim))
    #     # agents_ini = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    #     return agents_ini
    def generate_agent(self, species_num):
        # 强制定义三种角色的能力向量
        # Type 1: [1, 0, 0] Inspector
        # Type 2: [0, 1, 0] Manipulator
        # Type 3: [0, 0, 1] Sentinel
        agents_ini = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        # 如果 species_num > 3, 可以循环填充或报错，这里假设参数控制为3
        return agents_ini[:species_num]

    def generate_env(self):
        tasks_num = self.random_int(self.tasks_range[0], self.tasks_range[1] + 1)
        species_num = self.random_int(self.species_range[0], self.species_range[1] + 1)
        agents_species_num = [self.random_int(self.per_species_range[0], self.per_species_range[1] + 1) for _ in range(species_num)]

        agents_ini = self.generate_agent(species_num)
        tasks_ini = self.generate_task(tasks_num)
        while not np.all(np.matmul(agents_species_num, agents_ini) >= tasks_ini):
            agents_ini = self.generate_agent(species_num)
            tasks_ini = self.generate_task(tasks_num)

        depot_loc = self.random_value(species_num, 2)
        cost_ini = [self.random_value(1, 1) for _ in range(species_num)]
       # ... (在 generate_env 函数内部，找到生成 tasks_loc 的循环) ...
        
        # 替换原有的位置生成和死线逻辑
        tasks_loc = np.zeros((tasks_num, 2))
        deadlines = [] 

        for i in range(tasks_num):
            req_sum = int(np.sum(tasks_ini[i]))
            
            # --- 修改开始 ---
            
            # 1. Task 1 (简单任务, req_sum == 1)
            if req_sum == 1: 
                # Zone A: 靠近出生点 (0.0 - 0.3)
                tasks_loc[i, 0] = self.rng.uniform(0.0, 0.3)
                # 死线: 200s (完全不急，随便做)
                deadlines.append(200.0) 
            
            # 2. Task 2 (中等任务, req_sum == 2) -> 新增处理逻辑
            elif req_sum == 2:
                # Zone B: 中间区域 (0.4 - 0.7)
                tasks_loc[i, 0] = self.rng.uniform(0.4, 0.7)
                # 死线: 60s (有点急，贪心算法如果不先做这个，可能也会超时)
                deadlines.append(60.0) 

            # 3. Task 3 (困难任务, req_sum == 3)
            else: # req_sum >= 3
                # Zone C: 最远端 (0.8 - 1.0)
                tasks_loc[i, 0] = self.rng.uniform(0.8, 1.0)
                # 死线: 48s (十万火急，必须优先处理)
                deadlines.append(48.0) 
            
            # Y轴依然随机分布
            tasks_loc[i, 1] = self.rng.uniform(0, 1.0)
            
            # --- 修改结束 ---


        
        tasks_time = self.random_value(tasks_num, 1) * self.duration_scale

        task_dic = dict()
        agent_dic = dict()
        depot_dic = dict()
        species_dict = dict()
        species_dict['abilities'] = agents_ini
        species_dict['number'] = agents_species_num

        for i in range(tasks_num):
            task_dic[i] = {'ID': i,
                           'requirements': tasks_ini[i, :],  # requirements of the task
                           'members': [],  # members of the task
                           'cost': [],  # cost of each agent
                           'location': tasks_loc[i, :],  # location of the task
                           'feasible_assignment': False,  # whether the task assignment is feasible
                           'finished': False,
                           'failed': False,       # <--- 【新增】标记是否失败
                           'deadline': deadlines[i], # <--- 【新增】写入死线
                           'time_start': 0,
                           'time_finish': 0,
                           'status': tasks_ini[i, :],
                           'time': float(tasks_time[i, :]),
                           'sum_waiting_time': 0,
                           'efficiency': 0,
                           'abandoned_agent': [],
                           'optimized_ability': None,
                           'optimized_species': []}

        i = 0
        for s, n in enumerate(agents_species_num):
            species_dict[s] = []
            # 设定速度差异
            if s == 0:   # Type 1 自由巡检
                current_vel = 0.5  # 飞得快
            elif s == 1: # Type 2 操作机器
                current_vel = 0.1  # 爬得慢
            else:        # Type 3 环境感知
                current_vel = 0.3  # 速度中等

            for j in range(n):
                agent_dic[i] = {'ID': i,
                                'species': s,
                                'abilities': agents_ini[s, :],
                                'location': depot_loc[s, :],
                                'route': [- s - 1],
                                'current_task': - s - 1,
                                'contributed': False,
                                'arrival_time': [0.],
                                'cost': cost_ini[s],
                                'travel_time': 0,
                                'velocity': 0.2,
                                'next_decision': 0,
                                'depot': depot_loc[s, :],
                                'travel_dist': 0,
                                'sum_waiting_time': 0,
                                'current_action_index': 0,
                                'decision_step': 0,
                                'task_waiting_ratio': 1,
                                'trajectory': [],
                                'angle': 0,
                                'returned': False,
                                'assigned': False,
                                'pre_set_route': None,
                                'no_choice': False}
                species_dict[s].append(i)
                i += 1

        for s in range(species_num):
            depot_dic[s] = {'location': depot_loc[s, :],
                            'members': species_dict[s],
                            'ID': - s - 1}

        return task_dic, agent_dic, depot_dic, species_dict

    def generate_distance_matrix(self):
        species_distance_matrix = {}
        species_neighbor_matrix = {}
        for species in range(len(self.species_dict['number'])):
            tmp_dic = {-1: self.depot_dic[species], **self.task_dic}
            distances = {}
            for from_counter, from_node in tmp_dic.items():
                distances[from_counter] = {}
                for to_counter, to_node in tmp_dic.items():
                    if from_counter == to_counter:
                        distances[from_counter][to_counter] = 0
                    else:
                        distances[from_counter][to_counter] = self.calculate_eulidean_distance(from_node, to_node)

            sorted_distance_matrix = {k: sorted(dist, key=lambda x: dist[x]) for k, dist in distances.items()}
            species_distance_matrix[species] = distances
            species_neighbor_matrix[species] = sorted_distance_matrix
        return species_distance_matrix, species_neighbor_matrix

    def reset(self, test_env=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = None
        if test_env is not None:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = test_env
        else:
            self.task_dic, self.agent_dic, self.depot_dic, self.species_dict = self.generate_env()
        self.tasks_num = len(self.task_dic)
        self.agents_num = len(self.agent_dic)
        self.species_num = len(self.species_dict['number'])
        self.coalition_matrix = np.zeros((self.agents_num, self.tasks_num))
        self.current_time = 0
        self.finished = False

    def init_state(self):
        for task in self.task_dic.values():
            task.update(members=[], cost=[], finished=False, status=task['requirements'],feasible_assignment=False,
                        time_start=0, time_finish=0, sum_waiting_time=0, efficiency=0, abandoned_agent=[])
        for agent in self.agent_dic.values():
            agent.update(route=[-agent['species'] - 1], location=agent['depot'], contributed=False,
                         next_decision=0, travel_time=0, travel_dist=0, arrival_time=[0.], assigned=False,
                         sum_waiting_time=0, current_action_index=0, decision_step=0, trajectory=[], angle=0,
                         returned=False, pre_set_route=None, current_task=-1, task_waiting_ratio=1, no_choice=False, next_action=0)
        for depot in self.depot_dic.values():
            depot.update(members=self.species_dict[-depot['ID'] - 1])
        self.current_time = 0
        self.max_waiting_time = 200
        self.finished = False

    @staticmethod
    def find_by_key(data, target):
        for key, value in data.items():
            if isinstance(value, dict):
                yield from TaskEnv.find_by_key(value, target)
            elif key == target:
                yield value

    @staticmethod
    def get_matrix(dictionary, key):
        """
        :param key: the key to index
        :param dictionary: the dictionary for key to index
        """
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    @staticmethod
    def calculate_eulidean_distance(agent, task):
        return np.linalg.norm(agent['location'] - task['location'])

    def calculate_optimized_ability(self):
        for task in self.task_dic.values():
            task_status = task['status']
            # find all possible combinations of the group
            in_species_num = self.species_dict['number']
            species_ability = self.species_dict['abilities']
            num_set = [list(range(0, self.max_task_size + 1)) for _ in in_species_num]
            group_combinations = list(product(*num_set))

            abilities = []
            contained_spe = []
            for sample in group_combinations:
                ability = np.zeros((1, self.traits_dim))
                for spe, num in enumerate(sample):
                    ability += sample[spe] * species_ability[spe]
                contained_spe.append(np.array(sample) > 0)
                abilities.append(ability)

            effective_ability = np.maximum(np.minimum(task_status, np.vstack(abilities)), 0)
            score = np.divide(effective_ability, np.vstack(abilities), where=np.vstack(abilities) > 0,
                              out=np.zeros_like(np.vstack(abilities), dtype=float)) * effective_ability
            score = np.sum(score, axis=1)
            action_index = np.argmax(score)
            group_sort = np.argsort(score)[-2:]
            task['optimized_ability'] = abilities[action_index]
            optimized_species = []
            for ind in group_sort:
                optimized_species.append(contained_spe[ind])
            task['optimized_species'] = np.logical_or(*optimized_species)
        species_mask = np.vstack(self.get_matrix(self.task_dic, 'optimized_species'))
        return species_mask

    def get_current_agent_status(self, agent):
        status = []
        for a in self.agent_dic.values():
            if a['current_task'] >= 0:
                current_task = a['current_task']
                arrival_time = self.get_arrival_time(a['ID'], current_task)
                travel_time = np.clip(arrival_time - self.current_time, a_min=0, a_max=None)
                if self.current_time <= self.task_dic[current_task]['time_start']:
                    current_waiting_time = np.clip(self.current_time - arrival_time, a_min=0, a_max=None)
                    remaining_working_time = np.clip(self.task_dic[current_task]['time_start'] + self.task_dic[current_task]['time'] - self.current_time, a_min=0, a_max=None)
                else:
                    current_waiting_time = 0
                    remaining_working_time = 0
            else:
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            temp_status = np.hstack([a['abilities'], travel_time, remaining_working_time, current_waiting_time,
                                     agent['location'] - a['location'], a['assigned']])
            status.append(temp_status)
        current_agents = np.vstack(status)
        return current_agents

    def get_current_task_status(self, agent):
        status = []
        for t in self.task_dic.values():
            travel_time = self.calculate_eulidean_distance(agent, t) / agent['velocity']
            temp_status = np.hstack([t['status'], t['requirements'], t['time'], travel_time,
                                     agent['location'] - t['location'], t['feasible_assignment']])
            status.append(temp_status)
        status = [np.hstack([np.zeros(self.traits_dim), - np.ones(self.traits_dim), 0,
                             self.calculate_eulidean_distance(agent,
                                                              self.depot_dic[agent['species']])
                             / agent['velocity'], agent['location'] - agent['depot'], 1])] + status
        current_tasks = np.vstack(status)
        return current_tasks

    def get_unfinished_task_mask(self):
        mask = np.logical_not(self.get_unfinished_tasks())
        return mask

    def get_unfinished_tasks(self):
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(task['feasible_assignment'] is False and np.any(task['status'] > 0))
        return unfinished_tasks

    def get_arrival_time(self, agent_id, task_id):
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def get_abilities(self, members):
        if len(members) == 0:
            return np.zeros(self.traits_dim)
        else:
            return np.sum(np.array([self.agent_dic[member]['abilities'] for member in members]), axis=0)

    def get_contributable_task_mask(self, agent_id):
        agent = self.agent_dic[agent_id]
        contributable_task_mask = np.ones(self.tasks_num, dtype=bool)
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                ability = np.maximum(np.minimum(task['status'], agent['abilities']), 0.)
                if ability.sum() > 0:
                    contributable_task_mask[task['ID']] = False
        return contributable_task_mask
    # ==========================================================
    # 【新增：HRL 创新点3 - 动态掩码矩阵实时构造】
    # 将 Zone A/B/C 的严苛准入规则转化为网络能读懂的 Bool 矩阵
    # ==========================================================
    def get_zone_access_mask(self, agent_id):
        agent = self.agent_dic[agent_id]
        species = agent['species'] # 0: Type1(巡检), 1: Type2(操作), 2: Type3(感知)
        
        # 掩码逻辑：True 代表“屏蔽/禁止进入”，False 代表“开放/可进入”
        zone_mask = np.ones(self.tasks_num, dtype=bool) 
        
        for task in self.task_dic.values():
            x_loc = task['location'][0]
            task_id = task['ID']
            
            if x_loc < 0.4:
                # Zone A (物资仓储舱): 空间狭窄，仅允许 Type 1 进入
                if species == 0:
                    zone_mask[task_id] = False
            elif x_loc < 0.8:
                # Zone B (能源动力舱): 危险辐射，仅允许具备防护的 Type 2 和 Type 3 进入
                if species in [1, 2]:
                    zone_mask[task_id] = False
            else:
                # Zone C (核心控制舱): 开放区域，全员允许进入
                zone_mask[task_id] = False
                
        return zone_mask

    def get_waiting_tasks(self):
        waiting_tasks = np.ones(self.tasks_num, dtype=bool)
        waiting_agents = []
        for task in self.task_dic.values():
            if not task['feasible_assignment'] and len(task['members']) > 0:
                waiting_tasks[task['ID']] = False
                waiting_agents += task['members']
        return waiting_tasks, waiting_agents

    def agent_update(self):
        for agent in self.agent_dic.values():
            if agent['current_task'] < 0:
                if np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    agent['next_decision'] = np.nan
                elif not np.isnan(agent['next_decision']):
                    agent['next_decision'] = np.inf
                else:
                    pass
            else:
                current_task = self.task_dic[agent['current_task']]
                if current_task['feasible_assignment']:
                    if agent['ID'] in current_task['members']:
                        agent['next_decision'] = float(current_task['time_finish'])
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                        agent['assigned'] = False
                else:
                    agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                    agent['assigned'] = False

    def task_update(self):
        f_task = []
        # check each task status and whether it is finished
        for task in self.task_dic.values():
            # 如果已经结束或失败，跳过
            if task['finished'] or task.get('failed', False):
                continue
            
            # 【核心修改：超时判决】
            # 如果当前时间 > 截止时间，且任务还没开始干活 -> 判失败！
            if self.current_time > task['deadline'] and not task['feasible_assignment']:
                task['failed'] = True   # 标记失败
                task['finished'] = True # 逻辑上结束了（虽然没做完）
                # 可选：打印日志让你看到贪心算法是怎么死的
                # print(f"❌ Task {task['ID']} FAILED at {self.current_time:.1f}s (Deadline: {task['deadline']})")
                continue
            if not task['feasible_assignment']:
                abilities = self.get_abilities(task['members'])
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
                task['status'] = task['requirements'] - abilities  # update task status
                # Agents will wait for the other agents to arrive
                if (task['status'] <= 0).all():
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival, keepdims=True))
                        task['time_finish'] = float(np.max(arrival, keepdims=True) + task['time'])
                        task['feasible_assignment'] = True
                        f_task.append(task['ID'])
                    else:
                        task['feasible_assignment'] = False
                        infeasible_members = arrival <= np.max(arrival, keepdims=True) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
                else:
                    task['feasible_assignment'] = False
                    for member in np.array(task['members']):
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
                            task['abandoned_agent'].append(member)
            else:
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        # check depot status
        for depot in self.depot_dic.values():
            for member in depot['members']:
                if self.current_time >= self.get_arrival_time(member, depot['ID']) and np.all(self.get_matrix(self.task_dic, 'feasible_assignment')):
                    self.agent_dic[member]['returned'] = True
        return f_task

    def next_decision(self):
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            return ([], []), max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))
        no_choice = self.get_matrix(self.agent_dic, 'no_choice')
        decision_time = np.where(no_choice, np.inf, decision_time)
        next_decision = np.nanmin(decision_time)
        if np.isinf(next_decision):
            arrival_time = np.array([agent['arrival_time'][-1] for agent in self.agent_dic.values()])
            decision_time = np.where(no_choice, np.inf, arrival_time)
            next_decision = np.nanmin(decision_time)
        finished_agents = np.where(decision_time == next_decision)[0].tolist()
        blocked_agents = []
        for agent_id in np.where(np.isinf(decision_time))[0].tolist():
            if next_decision >= self.agent_dic[agent_id]['arrival_time'][-1]:
                blocked_agents.append(agent_id)
        release_agents = (finished_agents, blocked_agents)
        return release_agents, next_decision

    def agent_step(self, agent_id, task_id, decision_step):
        """
        :param agent_id: the id of agent
        :param task_id: the id of task
        :param decision_step: the decision step of the agent
        :return: end_episode, finished_tasks
        """
        #  choose any task
        task_id = task_id - 1
        if task_id != -1:
            agent = self.agent_dic[agent_id]
            task = self.task_dic[task_id]
            if task['feasible_assignment']:
                return -1, False, []
        else:
            agent = self.agent_dic[agent_id]
            task = self.depot_dic[agent['species']]
        agent['route'].append(task['ID'])
        previous_task = agent['current_task']
        agent['current_task'] = task_id
        travel_time = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['travel_time'] = travel_time
        agent['travel_dist'] += self.calculate_eulidean_distance(agent, task)
        if previous_task >= 0 and self.task_dic[previous_task]['time_finish'] < self.current_time:
            current_time = self.task_dic[previous_task]['time_finish']
        else:
            current_time = self.current_time
        agent['arrival_time'] += [current_time + travel_time]
        # calculate the angle from current location to next location
        agent['location'] = task['location']
        agent['decision_step'] = decision_step
        agent['no_choice'] = False

        if agent_id not in task['members']:
            task['members'].append(agent_id)
        f_t = self.task_update()
        self.agent_update()
        return 0, True, f_t

    # ======== 【修改 2：多重硬约束过滤层叠加】 ========
    def agent_observe(self, agent_id, max_waiting=False):
        agent = self.agent_dic[agent_id]
        
        # 1. 基础掩码：屏蔽已经完成的任务
        mask = self.get_unfinished_task_mask()
        
        # 2. 异构技能匹配掩码 (C_skill)：屏蔽自己能力无法满足的任务
        contributable_mask = self.get_contributable_task_mask(agent_id)
        
        # 3. 区域准入权限掩码 (C_access)：屏蔽物理隔离舱段的任务
        access_mask = self.get_zone_access_mask(agent_id)
        
        # 聚合所有掩码 (逻辑或：只要有一项违规，该动作即被判定为非法)
        mask = np.logical_or(mask, contributable_mask)
        mask = np.logical_or(mask, access_mask) 
        
        if max_waiting:
            waiting_tasks_mask, waiting_agents = self.get_waiting_tasks()
            waiting_len = np.sum(waiting_tasks_mask == 0)
            if waiting_len > 5:
                mask = np.logical_or(mask, waiting_tasks_mask)
                
        # 索引0代表 Depot (停泊点)，永远合法 (False代表不屏蔽)
        mask = np.insert(mask, 0, False)
        
        agents_info = np.expand_dims(self.get_current_agent_status(agent), axis=0)
        tasks_info = np.expand_dims(self.get_current_task_status(agent), axis=0)
        mask = np.expand_dims(mask, axis=0)
        return tasks_info, agents_info, mask

    def calculate_waiting_time(self):
        for agent in self.agent_dic.values():
            agent['sum_waiting_time'] = 0
        for task in self.task_dic.values():
            arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
            if len(arrival) != 0:
                if task['feasible_assignment']:
                    task['sum_waiting_time'] = np.sum(np.max(arrival) - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
                else:
                    task['sum_waiting_time'] = np.sum(self.current_time - arrival) \
                                               + len(task['abandoned_agent']) * self.max_waiting_time
            else:
                task['sum_waiting_time'] = len(task['abandoned_agent']) * self.max_waiting_time
            for member in task['members']:
                if task['feasible_assignment']:
                    self.agent_dic[member]['sum_waiting_time'] += np.max(arrival) - self.get_arrival_time(member, task['ID'])
                else:
                    self.agent_dic[member]['sum_waiting_time'] += self.current_time - self.get_arrival_time(member, task['ID']) if self.current_time - self.get_arrival_time(member, task['ID']) > 0 else 0
            for member in task['abandoned_agent']:
                self.agent_dic[member]['sum_waiting_time'] += self.max_waiting_time

    def check_finished(self):
        self.task_update()
        decision_agents, current_time = self.next_decision()
        # dead_lock = self.check_dead_lock()
        if len(decision_agents[0]) + len(decision_agents[1]) == 0:
            self.current_time = current_time
            finished = np.all(self.get_matrix(self.agent_dic, 'returned')) and np.all(self.get_matrix(self.task_dic, 'finished'))
        else:
            finished = False
        return finished

    def generate_traj(self):
        for agent in self.agent_dic.values():
            # save the location of the agent as trajectory
            time_step = 0
            angle = 0
            for i in range(1, len(agent['route'])):
                current_task = self.task_dic[agent['route'][i - 1]] if agent['route'][i - 1] >= 0 else self.depot_dic[agent['species']]
                next_task = self.task_dic[agent['route'][i]] if agent['route'][i] >= 0 else self.depot_dic[agent['species']]
                angle = np.arctan2(next_task['location'][1] - current_task['location'][1],
                                   next_task['location'][0] - current_task['location'][0])
                distance = self.calculate_eulidean_distance(next_task, current_task)
                total_time = distance / agent['velocity']
                arrival_time_next = agent['arrival_time'][i]
                arrival_time_current = agent['arrival_time'][i - 1]
                if next_task['ID'] >= 0 and agent['ID'] in next_task['members'] \
                        and next_task['feasible_assignment']:
                    if next_task['time_start'] - arrival_time_next <= self.max_waiting_time:
                        next_decision = next_task['time_finish']
                    else:
                        next_decision = arrival_time_next + self.max_waiting_time
                elif next_task['ID'] < 0 and i != len(agent['route']) - 1:
                    next_decision = arrival_time_next + self.depot_waiting_time
                else:
                    next_decision = arrival_time_next + self.max_waiting_time
                if current_task['ID'] < 0 and i == 1:
                    current_decision = 0
                elif current_task['ID'] < 0:
                    current_decision = arrival_time_current + self.depot_waiting_time
                else:
                    if agent['ID'] in current_task['members'] \
                            and current_task['time_start'] - arrival_time_current <= self.max_waiting_time \
                            and current_task['feasible_assignment']:
                        current_decision = current_task['time_finish']
                    else:
                        current_decision = arrival_time_current + self.max_waiting_time
                while time_step < next_decision:
                    time_step += self.dt
                    if time_step < arrival_time_next:
                        fraction_of_time = (time_step - current_decision) / total_time
                        if fraction_of_time <= 1:
                            x = current_task['location'][0] + fraction_of_time * (
                                        next_task['location'][0] - current_task['location'][0])
                            y = current_task['location'][1] + fraction_of_time * (
                                        next_task['location'][1] - current_task['location'][1])
                            agent['trajectory'].append(np.hstack([x, y, angle]))
                        else:
                            agent['trajectory'].append(np.hstack([next_task['location'][0], next_task['location'][1], angle]))
                    else:
                        agent['trajectory'].append(np.array([next_task['location'][0], next_task['location'][1], angle]))
            while time_step < self.current_time:
                time_step += self.dt
                agent['trajectory'].append(np.array([self.depot_dic[agent['species']]['location'][0], self.depot_dic[agent['species']]['location'][1], angle]))

    def get_episode_reward(self, max_time=100):
        self.calculate_waiting_time()
        eff = self.get_efficiency()
        
        # 统计成功和失败
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        failed_tasks = [t.get('failed', False) for t in self.task_dic.values()]
        
        success_count = np.sum(finished_tasks) - np.sum(failed_tasks)
        fail_count = np.sum(failed_tasks)
        
        # 【核心修改：奖励函数】
        # 基础奖励：完工时间越短越好
        base_reward = - self.current_time - eff * 10
        
        # 惩罚项：死掉一个任务扣 200 分！(迫使RL必须学会保成功率)
        penalty = fail_count * 200 
        
        final_reward = base_reward - penalty
        
        # 注意：为了让外部代码能统计成功率，我们这里的 finished_tasks 需要通过逻辑处理
        # 或者我们直接返回一个特定的 info。但为了兼容旧代码，这里不需要大改返回结构。
        # 只要确保 reward 足够低，RL 就会学乖。
        
        return final_reward, finished_tasks

    def get_efficiency(self):
        for task in self.task_dic.values():
            if task['feasible_assignment']:
                task['efficiency'] = abs(np.sum(task['requirements'] - task['status'])) / task['requirements'].sum()
            else:
                task['efficiency'] = 10
        efficiency = np.mean(self.get_matrix(self.task_dic, 'efficiency'))
        return efficiency

    def stack_trajectory(self):
        for agent in self.agent_dic.values():
            agent['trajectory'] = np.vstack(agent['trajectory'])

    def plot_animation(self, path, n):
        self.generate_traj()
        plot_robot_icon = False  # 定义缺失变量
        
        # 颜色映射
        def get_cmap(n, name='Dark2'):
            return plt.cm.get_cmap(name, n)
        cmap = get_cmap(self.species_num)

        self.stack_trajectory()
        finished_tasks = self.get_matrix(self.task_dic, 'finished')
        finished_rate = np.sum(finished_tasks) / len(finished_tasks)
        gif_len = int(self.current_time/self.dt)

        # --- 初始化画布 ---
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, aspect='equal')
        
        # 绘制分区线
        ax.axvline(x=3.3, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        ax.axvline(x=6.6, color='gray', linestyle=':', linewidth=2, alpha=0.5)
        
        # 绘制区域文字
        top_y = 10.5
        ax.text(1.65, top_y, "Zone A\n(Storage)", ha='center', fontsize=12, fontweight='bold', color='#D62728', alpha=0.8)
        ax.text(5.0, top_y, "Zone B\n(Energy/High-Risk)", ha='center', fontsize=12, fontweight='bold', color='#FF7F0E', alpha=0.8)
        ax.text(8.35, top_y, "Zone C\n(Control)", ha='center', fontsize=12, fontweight='bold', color='#9467BD', alpha=0.8)

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 11.5)
        ax.axis('off') # 隐藏坐标轴，更美观

        # 初始化线条
        lines = [ax.plot([], [], color=cmap(a['species']), alpha=0.6, linewidth=1)[0] for a in self.agent_dic.values()]

        # 绘制基地 (黑色方块)
        for d in self.depot_dic.values():
            ax.add_patch(patches.Rectangle(
                xy=(d['location'][0]*10-0.25, d['location'][1]*10-0.25),
                width=0.5, height=0.5, color='black', alpha=0.7, zorder=5
            ))

        # 绘制任务点 (保存引用以便 update 更新)
        task_patches = {}
        for task in self.task_dic.values():
            req_sum = int(task['requirements'].sum())
            # 颜色映射：T1红, T2橙, T3紫
            if req_sum == 1: c = '#D62728' # Red
            elif req_sum == 2: c = '#FF7F0E' # Orange
            else: c = '#9467BD' # Purple
            
            # 形状：4/5/6边形
            patch = ax.add_patch(patches.RegularPolygon(
                xy=(task['location'][0]*10, task['location'][1]*10),
                numVertices=req_sum + 3,
                radius=0.35,
                facecolor=c,
                edgecolor='black',
                linewidth=0.8,
                alpha=0.9,
                zorder=10 # 保证任务在最上层
            ))
            task_patches[task['ID']] = patch

        # 初始化机器人 (三角形)
        agent_patches = []
        agent_texts = []
        for a in self.agent_dic.values():
            start_loc = self.depot_dic[a['species']]['location']
            # 机器人图标
            tri = ax.add_patch(patches.RegularPolygon(
                xy=(start_loc[0]*10, start_loc[1]*10),
                numVertices=3, radius=0.3, color=cmap(a['species']), zorder=20
            ))
            agent_patches.append(tri)
            # 机器人ID
            txt = ax.text(start_loc[0]*10, start_loc[1]*10, str(a['ID']),
                          ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=21)
            agent_texts.append(txt)

        # 图例 (Legend)
        legend_elements = [
            Line2D([0], [0], color='gray', linestyle=':', label='Zone Boundary'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#D62728', label='Task 1 (Insp)', markersize=10),
            Line2D([0], [0], marker='p', color='w', markerfacecolor='#FF7F0E', label='Task 2 (Maint)', markersize=10),
            Line2D([0], [0], marker='h', color='w', markerfacecolor='#9467BD', label='Task 3 (Emerg)', markersize=10),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=cmap(0), label='Type 1 Robot', markersize=10),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=cmap(1), label='Type 2 Robot', markersize=10),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=cmap(2), label='Type 3 Robot', markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

        # --- 动画更新逻辑 ---
        def update(frame):
            # 标题显示进度
            ax.set_title(f'Sim Time: {frame * self.dt:.1f} min | Progress: {finished_rate:.0%}', fontsize=14)
            
            # 1. 更新机器人位置
            for i, agent in self.agent_dic.items():
                if frame < len(agent['trajectory']):
                    pos = agent['trajectory'][frame]
                    # 更新三角形
                    agent_patches[i].xy = (pos[0]*10, pos[1]*10)
                    agent_patches[i].orientation = pos[2] - np.pi/2
                    # 更新文字
                    agent_texts[i].set_position((pos[0]*10, pos[1]*10))
                    # 更新轨迹线 (只保留最近30帧，避免画面太乱)
                    start_t = max(0, frame - 30)
                    lines[i].set_data(
                        agent['trajectory'][start_t:frame+1, 0]*10, 
                        agent['trajectory'][start_t:frame+1, 1]*10
                    )

            # 2. 更新任务状态 (只变绿，不通过 set_color('w') 变灰)
            for t_id, task in self.task_dic.items():
                # 如果任务完成了，且当前时间超过完成时间 -> 变绿
                if task['finished'] and (frame * self.dt >= task['time_finish']):
                    task_patches[t_id].set_facecolor('#2CA02C') # Green
                    task_patches[t_id].set_edgecolor('#006400') # Dark Green Border
                # 注意：我们删除了 else: set_color('w') 的逻辑，这样未完成的任务会保持红/橙/紫

            return lines + agent_patches + list(task_patches.values())

        # 保存
        ani = FuncAnimation(fig, update, frames=gif_len, interval=100, blit=False)
        ani.save(f'{path}/{n}_Time_{self.current_time:.1f}.gif', writer='pillow')

        
    def execute_by_route(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        self.max_waiting_time = 200
        while not self.finished and self.current_time < 200:
            decision_agents, current_time = self.next_decision()
            self.current_time = current_time
            decision_agents = decision_agents[0] + decision_agents[1]
            for agent in decision_agents:
                if self.agent_dic[agent]['pre_set_route'] is None or not self.agent_dic[agent]['pre_set_route']:
                    self.agent_step(agent, 0, 0)
                    self.agent_dic[agent]['next_decision'] = np.nan
                    continue
                self.agent_step(agent, self.agent_dic[agent]['pre_set_route'].pop(0), 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def execute_greedy_action(self, path='./', method=0, plot_figure=False):
        self.plot_figure = plot_figure
        while not self.finished and self.current_time < 200:
            release_agents, current_time = self.next_decision()
            self.current_time = current_time
            while release_agents[0] or release_agents[1]:
                agent_id = release_agents[0].pop(0) if release_agents[0] else release_agents[1].pop(0)
                agent = self.agent_dic[agent_id]
                tasks_info, agents_info, mask = self.agent_observe(agent_id, max_waiting=True)
                dist = np.inf
                action = None
                for task_id, masked in enumerate(mask[0, :]):
                    if not masked:
                        dist_ = self.calculate_eulidean_distance(agent, self.task_dic[
                            task_id - 1]) if task_id - 1 >= 0 else self.calculate_eulidean_distance(agent,
                                                                                          self.depot_dic[agent['species']])
                        if dist_ < dist:
                            action = task_id
                self.agent_step(agent_id, action, 0)
            self.finished = self.check_finished()
        if self.plot_figure:
            self.plot_animation(path, method)
        print(self.current_time)
        return self.current_time

    def pre_set_route(self, routes, agent_id):
        if not self.agent_dic[agent_id]['pre_set_route']:
            self.agent_dic[agent_id]['pre_set_route'] = routes
        else:
            self.agent_dic[agent_id]['pre_set_route'] += routes

    def process_map(self, path):
        import pandas as pd
        grouped_tasks = dict()
        groups = list(set(np.array(self.get_matrix(self.task_dic, 'requirements')).squeeze(1).tolist()))
        for task_requirement in groups:
            grouped_tasks[task_requirement] = dict()
        index = np.zeros_like(groups)
        for i, task in self.task_dic.items():
            requirement = int(task['requirements'])
            ind = index[groups.index(requirement)]
            grouped_tasks[requirement].update({ind: task})
            index[groups.index(requirement)] += 1
        grouped_tasks = {key: value for key, value in grouped_tasks.items() if len(value) > 0}
        time_finished = [self.get_matrix(dic, 'time_finish') for dic in grouped_tasks.values()]
        t = 0
        time_tick_stamp = dict()
        while t <= self.current_time:
            time_tick_stamp[t] = [np.sum(np.array(ratio) < t)/len(ratio) for ratio in time_finished]
            t += 0.1
            t = np.round(t, 1)
        pd = pd.DataFrame(time_tick_stamp)
        pd.to_csv(f'{path}time_RL.csv')


# if __name__ == '__main__':
#     import pickle
#     testSet = 'RALTestSet'
#     os.mkdir(f'../{testSet}')
#     for i in range(50):
#         env = TaskEnv((3, 3), (5, 5), (20, 20), 5, seed=i)
#         pickle.dump(env, open(f'../{testSet}/env_{i}.pkl', 'wb'))
#     env.init_state()
if __name__ == '__main__':
    import pickle
    import os

    # 1. 定义测试集文件夹名字
    testSet = 'SpaceStationTestSet/3_11_FGH_TestSet'
    
    # 2. 确保路径是在当前目录下 (去掉 ../ 改为 ./)
    save_path = f'./{testSet}' 
    
    # 3. 创建文件夹 (exist_ok=True 防止文件夹已存在时报错)
    os.makedirs(save_path, exist_ok=True)
    
    print(f"正在生成测试集，请稍候...")
    
    for i in range(50):
        # 参数要和 parameters.py 中的 EnvParams 保持一致 (Traits=3)
        env = HierarchicalTaskEnv(
            per_species_range=(2, 2), 
            species_range=(3, 3), 
            tasks_range=(15, 15), 
            traits_dim=3, 
            seed=i
        )
        # 保存到指定路径
        pickle.dump(env, open(f'{save_path}/env_{i}.pkl', 'wb'))
        
    print(f"✅ 生成完毕！")
    print(f"测试集已保存在: {os.path.abspath(save_path)}")