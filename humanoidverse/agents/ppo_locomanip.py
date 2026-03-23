import torch  # 导入 PyTorch 主包
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic  # 导入 PPO 策略和价值网络
from humanoidverse.agents.modules.data_utils import RolloutStorage  # 导入 rollout 存储工具
from humanoidverse.envs.base_task.base_task import BaseTask  # 导入环境基类
from humanoidverse.agents.base_algo import BaseAlgo  # 导入算法基类
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback  # 导入评估回调
from humanoidverse.utils.average_meters import TensorAverageMeterDict  # 导入张量统计工具

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter  # Tensorboard 日志工具
import time  # 时间工具
import os  # 操作系统工具
import statistics  # 统计工具
from collections import deque  # 双端队列，用于滑动窗口统计
from hydra.utils import instantiate  # Hydra 配置动态实例化
from loguru import logger  # 日志工具
from rich.progress import track  # 进度条工具
from rich.console import Console  # 富文本控制台
from rich.panel import Panel  # 富文本面板
from rich.live import Live  # 实时刷新工具
console = Console()  # 初始化控制台

##################################################################
######## 注意：本 agent 仅用于评估目的，不用于训练 ########
##################################################################
class PPOLocoManip(BaseAlgo):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        # 初始化 PPO LocoManip agent
        self.device= device  # 运行设备
        self.env = env  # 环境对象
        self.config = config  # 配置参数
        self.log_dir = log_dir  # 日志目录

        self._init_config()  # 初始化配置

        self.tot_timesteps = 0  # 总步数计数
        self.tot_time = 0  # 总耗时计数

        # 记录相关变量
        self.ep_infos = []  # 每回合信息
        self.rewbuffer = deque(maxlen=100)  # 奖励滑动窗口
        self.lenbuffer = deque(maxlen=100)  # 回合长度滑动窗口
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前奖励和
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前回合长度

        self.eval_callbacks: list[RL_EvalCallback] = []  # 评估回调列表
        self.episode_env_tensors = TensorAverageMeterDict()  # 环境统计张量
        _ = self.env.reset_all()  # 重置所有环境

        # 两种策略的观测名（站立/行走）
        self.actor_obs = ['actor_stand_obs', 'actor_loco_obs']

    def _init_config(self):
        # 环境相关配置初始化
        self.num_envs: int = self.env.config.num_envs  # 并行环境数
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict  # 观测维度字典
        self.num_act = self.env.config.robot.lower_body_actions_dim  # 动作维度（下半身）

    def setup(self):
        # 初始化模型和优化器
        logger.info("Setting up PPO Loco Manip")
        self._setup_models_and_optimizer()

    def _setup_models_and_optimizer(self):
        # 初始化两个 actor（站立/行走）
        self.actor_stand = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor_stand,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)
        self.actor_loco = PPOActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor_loco,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std
        ).to(self.device)
        self.actor = [self.actor_stand, self.actor_loco]  # 两种 actor 组成列表

    def _eval_mode(self):
        # 设置为评估模式（禁用 dropout/bn）
        self.actor[0].eval()
        self.actor[1].eval()

    def load(self, ckpt_stand_path, ckpt_loco_path):
        # 加载两个 actor 的 checkpoint
        if ckpt_stand_path is not None:
            logger.info(f"Loading standing checkpoint from {ckpt_stand_path}")
            loaded_dict_stand = torch.load(ckpt_stand_path, map_location=self.device)
            self.actor[0].load_state_dict(loaded_dict_stand["actor_model_state_dict"])
        if ckpt_loco_path is not None:
            logger.info(f"Loading locomotion checkpoint from {ckpt_loco_path}")
            loaded_dict_loco = torch.load(ckpt_loco_path, map_location=self.device)
            self.actor[1].load_state_dict(loaded_dict_loco["actor_model_state_dict"])
        return loaded_dict_stand['infos'], loaded_dict_loco["infos"]  # 返回 checkpoint 附带信息

    def _process_env_step(self, rewards, dones, infos):
        # 处理环境步进后的操作（如重置 actor 状态）
        self.actor[self.env.control_mode].reset(dones)

    def _actor_act_step(self, obs_dict):
        return self.actor[self.env.control_mode].act(obs_dict)
    
    @property
    def inference_model(self):
        return self.actor

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################

    def env_step(self, actor_state):
        actions_lower_body = actor_state["actions"]
        actions_upper_body = self.env.ref_upper_dof_pos
        actions = torch.cat([actions_lower_body, actions_upper_body], dim=1)
        obs_dict, rewards, dones, extras = self.env.step(actions)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()
        for obs_key in obs_dict.keys():
            print(obs_key, sorted(self.env.config.obs.obs_dict[obs_key]))
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return obs_dict

    @torch.no_grad()
    def evaluate_loco_manip_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)
        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
            step += 1

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict):
        eval_policy = self.eval_policy[self.env.control_mode]
        actor_obs = actor_state["obs"][self.actor_obs[self.env.control_mode]]
        actions = eval_policy(actor_obs)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def _get_inference_policy(self, device=None):
        self._eval_mode()
        if device is not None:
            self.actor[0].to(device)
            self.actor[1].to(device)
        return [self.actor[0].act_inference,
                self.actor[1].act_inference]