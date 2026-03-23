import torch  # 导入 PyTorch 主包

from torch import nn, Tensor  # 导入常用的神经网络和张量类型

import time  # 时间相关工具
import math  # 数学工具
from pathlib import Path  # 路径处理
from typing import Optional, List, Tuple, Dict  # 类型注解

from hydra.utils import instantiate  # Hydra 配置工具，用于动态实例化对象

from humanoidverse.envs.base_task.base_task import BaseTask  # 导入环境基类

# 算法基类，所有 RL 算法都应继承自该类
class BaseAlgo:
    def __init__(self, env: BaseTask, config, device):
        self.env = env  # 训练/评估所用环境
        self.config = config  # 算法和实验配置
        self.device = device  # 运行设备（如 'cpu' 或 'cuda'）

    def setup(self):
        # 算法初始化方法，子类需实现
        return NotImplementedError

    def learn(self):
        # 算法主训练循环，子类需实现
        return NotImplementedError

    def load(self, path):
        # 加载模型参数，子类需实现
        return NotImplementedError
    
    @property
    def inference_model(self):
        # 返回推理用模型，子类需实现
        return NotImplementedError

    def env_step(self, actions, extra_info=None):
        # 与环境交互，执行一步动作
        obs_dict, rewards, dones, extras = self.env.step(actions, extra_info)
        return obs_dict, rewards, dones, extras  # 返回观测、奖励、终止标志和额外信息

    @torch.no_grad()  # 禁用梯度计算，加速推理
    def evaluate_policy(self):
        # 策略评估方法，子类需实现
        return NotImplementedError

    def save(self, path=None, name="last.ckpt"):
        # 保存模型参数，子类需实现
        raise NotImplementedError
