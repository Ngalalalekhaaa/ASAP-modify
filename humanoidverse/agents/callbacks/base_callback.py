import torch  # 导入 PyTorch 主包
from torch.nn import Module  # 导入神经网络模块基类

# RL 评估回调基类，所有自定义回调都应继承自该类
class RL_EvalCallback(Module):
    def __init__(self, config, training_loop):
        super().__init__()  # 初始化父类
        self.config = config  # 回调相关配置
        self.training_loop = training_loop  # 训练主循环对象
        self.device = self.training_loop.device  # 设备信息（如 'cpu' 或 'cuda'）

    def on_pre_evaluate_policy(self):
        # 策略评估前的钩子，可在子类中实现自定义逻辑
        pass

    def on_pre_eval_env_step(self, actor_state):
        # 环境步进前的钩子，可在子类中实现自定义逻辑
        return actor_state
    
    def on_post_eval_env_step(self, actor_state):
        # 环境步进后的钩子，可在子类中实现自定义逻辑
        return actor_state
    
    def on_post_evaluate_policy(self):
        # 策略评估后的钩子，可在子类中实现自定义逻辑
        pass
