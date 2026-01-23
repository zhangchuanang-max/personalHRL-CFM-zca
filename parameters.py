# 原来的代码
# class EnvParams:
#     SPECIES_AGENTS_RANGE = (3, 3)
#     SPECIES_RANGE = (3, 5)
#     TASKS_RANGE = (15, 50)
#     MAX_TIME = 200
#     TRAIT_DIM = 5
#     DECISION_DIM = 30

#新代码
class EnvParams:
    # 空间站场景设定
    SPECIES_AGENTS_RANGE = (2, 4) # 每种类型的机器人数量范围
    SPECIES_RANGE = (3, 3)        # 固定为3种角色 (Type 1, 2, 3)
    TASKS_RANGE = (20, 40)        # 任务数量
    MAX_TIME = 200                # 最大仿真时间
    TRAIT_DIM = 3                 # 核心修改：能力维度改为3 [视觉, 操作, 感知]
    DECISION_DIM = 30             # 根据任务数调整

class TrainParams:
    USE_GPU = True
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
    NUM_META_AGENT = 4
    LR = 1e-5
    GAMMA = 1
    DECAY_STEP = 2e3
    RESET_OPT = False
    EVALUATE = True
    EVALUATION_SAMPLES = 256
    RESET_RAY = False
    INCREASE_DIFFICULTY = 20000
    SUMMARY_WINDOW = 8
    DEMON_RATE = 0.5
    IL_DECAY = -1e-5  # -1e-6 700k decay 0.5, -1e-5 70k decay 0.5, -1e-4 7k decay 0.5
    BATCH_SIZE = 512
    AGENT_INPUT_DIM = 6 + EnvParams.TRAIT_DIM
    TASK_INPUT_DIM = 5 + 2 * EnvParams.TRAIT_DIM
    EMBEDDING_DIM = 128
    SAMPLE_SIZE = 200
    PADDING_SIZE = 50
    POMO_SIZE = 10
    FORCE_MAX_OPEN_TASK = True


class SaverParams:
    #FOLDER_NAME = 'save_1'     #原始情况的训练结果
    #FOLDER_NAME = 'save_ablation_no_seq'    #创新点1的消融实验（看不到队友情况）
    
    FOLDER_NAME = 'save_with_cfm'   # 建议命名：save_with_cfm (带有CFM机制) 或 save_innovation2
    MODEL_PATH = f'model/{FOLDER_NAME}'
    TRAIN_PATH = f'train/{FOLDER_NAME}'
    GIFS_PATH = f'gifs/{FOLDER_NAME}'
    LOAD_MODEL = True      #True能加载之前的模型
    #LOAD_MODEL = False  # 3. 【重置】我们要从头训练一个笨蛋模型，不能加载之前的
    LOAD_FROM = 'best'  # 'best'
    SAVE = True
    SAVE_IMG = True
    SAVE_IMG_GAP = 1000

