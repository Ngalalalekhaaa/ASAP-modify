import torch  # 导入 PyTorch 主包
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块

from humanoidverse.agents.modules.ppo_modules import PPOActorFixSigma  # 导入 PPO 策略网络（固定方差）
from humanoidverse.agents.modules.models import BaseModels  # 导入模型基类
from humanoidverse.agents.modules.data_utils import RolloutStorage  # rollout 存储工具
from humanoidverse.envs.base_task.base_task import BaseEnv  # 环境基类
from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # 算法基类
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback  # 评估回调基类
from humanoidverse.utils.average_meters import TensorAverageMeterDict  # 张量统计工具

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter  # Tensorboard 日志工具
import time  # 时间工具
import os  # 操作系统工具
import statistics  # 统计工具
from collections import deque  # 双端队列
from hydra.utils import instantiate  # Hydra 配置动态实例化
from loguru import logger  # 日志工具
from rich.progress import track  # 进度条工具
from rich.console import Console  # 富文本控制台
from rich.panel import Panel  # 富文本面板
from rich.live import Live  # 实时刷新工具
console = Console()  # 初始化控制台

# DAgger 算法 agent 类，实现模仿学习与 RL 结合的数据集聚合训练
class DAgger(BaseAlgo):
    def __init__(self,
                 env: BaseEnv,
                 config,
                 log_dir=None,
                 device='cpu'):
        # 初始化 DAgger agent
        self.device= device  # 运行设备
        self.env = env  # 环境对象
        self.config = config  # 配置参数
        self.log_dir = log_dir  # 日志目录
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)  # Tensorboard 日志

        self.start_time = 0  # 计时器：开始时间
        self.stop_time = 0  # 计时器：结束时间
        self.collection_time = 0  # 数据收集耗时
        self.learn_time = 0  # 学习耗时

        self._init_config()  # 初始化配置

        self.tot_timesteps = 0  # 总步数计数
        self.tot_time = 0  # 总耗时计数
        self.current_learning_iteration = 0  # 当前学习迭代数

        self.eval_callbacks: list[RL_EvalCallback] = []  # 评估回调列表
        self.episode_env_tensors = TensorAverageMeterDict()  # 环境统计张量
        _, _ = self.env.reset_all()  # 重置所有环境

    def _init_config(self):
        # 环境相关配置
        self.num_envs: int = self.env.config.num_envs  # 并行环境数
        self.algo_obs_dim_dict = self.env.config.robot.algo_obs_dim_dict  # 观测维度字典
        self.num_act = self.env.config.robot.actions_dim  # 动作维度

        # 日志相关配置
        self.save_interval = self.config.save_interval  # 保存间隔
        # 训练相关配置
        self.num_steps_per_env = self.config.num_steps_per_env  # 每个环境步数
        self.load_optimizer = self.config.load_optimizer  # 是否加载优化器
        self.num_learning_iterations = self.config.num_learning_iterations  # 总训练迭代数
        self.init_at_random_ep_len = self.config.init_at_random_ep_len  # 是否随机初始化回合长度

        # 算法相关配置
        self.desired_kl = self.config.desired_kl  # 目标 KL 散度
        self.schedule = self.config.schedule  # 学习率调度
        self.learning_rate = self.config.learning_rate  # 学习率
        self.clip_param = self.config.clip_param  # PPO 裁剪参数
        self.num_learning_epochs = self.config.num_learning_epochs  # 每次迭代的 epoch 数
        self.num_mini_batches = self.config.num_mini_batches  # 每次迭代的 mini-batch 数
        self.gamma = self.config.gamma  # 折扣因子
        self.lam = self.config.lam  # GAE 参数
        self.value_loss_coef = self.config.value_loss_coef  # 价值损失系数
        self.entropy_coef = self.config.entropy_coef  # 熵正则系数
        self.max_grad_norm = self.config.max_grad_norm  # 最大梯度范数
        self.use_clipped_value_loss = self.config.use_clipped_value_loss  # 是否裁剪价值损失

    def setup(self):
        # 初始化模型、优化器和存储
        logger.info("Setting up Dagger")        
        
        actor_network_dict = dict(actor=self.config.network_dict['actor'])  # 策略网络结构
        teacher_actor_network_dict = dict(actor=self.config.network_dict['teacher_actor'])  # 专家网络结构
        teacher_actor_network_load_dict = dict(actor=self.config.network_load_dict['teacher_actor'])  # 专家网络加载参数
        self.actor = PPOActorFixSigma(self.algo_obs_dim_dict, actor_network_dict, {}, self.num_act)  # 初始化 agent 策略
        self.gt_actor = PPOActorFixSigma(self.algo_obs_dim_dict, teacher_actor_network_dict, teacher_actor_network_load_dict, self.num_act)  # 初始化专家策略

        self.actor.to(self.device)  # 策略网络转到设备
        self.gt_actor.to(self.device)  # 专家网络转到设备
        # import ipdb; ipdb.set_trace()  # 调试断点
        # print all keys in self.actor.parameters()  # 打印参数名
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)  # 优化器
        logger.info(f"Setting up Storage")
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env)  # rollout 存储
        ## Register obs keys  # 注册观测键
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            self.storage.register_key(obs_key, shape=(obs_dim,), dtype=torch.float)  # 注册观测键及其形状
        
        ## Register others
        self.storage.register_key('actions', shape=(self.num_act,), dtype=torch.float)  # 注册动作
        self.storage.register_key('actions_log_prob', shape=(1,), dtype=torch.float)  # 注册动作对数概率
        self.storage.register_key('action_mean', shape=(self.num_act,), dtype=torch.float)  # 注册动作均值
        self.storage.register_key('action_sigma', shape=(self.num_act,), dtype=torch.float)  # 注册动作方差

        self.storage.register_key('gt_actions', shape=(self.num_act,), dtype=torch.float)  # 注册专家动作
        self.storage.register_key('gt_actions_log_prob', shape=(1,), dtype=torch.float)  # 注册专家动作对数概率
        self.storage.register_key('gt_action_mean', shape=(self.num_act,), dtype=torch.float)  # 注册专家动作均值
        self.storage.register_key('gt_action_sigma', shape=(self.num_act,), dtype=torch.float)  # 注册专家动作方差
        
        # Book keeping  # 记录相关变量
        self.ep_infos = []  # 每回合信息
        self.rewbuffer = deque(maxlen=100)  # 奖励滑动窗口
        self.lenbuffer = deque(maxlen=100)  # 回合长度滑动窗口
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前奖励和
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前回合长度
    
    def _eval_mode(self):
        self.actor.eval()  # 设置为评估模式

    def _train_mode(self):
        self.actor.train()  # 设置为训练模式

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")  # 加载 checkpoint
            loaded_dict = torch.load(ckpt_path)
            self.actor.load_state_dict(loaded_dict["actor_state_dict"])  # 加载 actor 参数
            if self.load_optimizer:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])  # 加载优化器参数
                self.learning_rate = loaded_dict['optimizer_state_dict']['param_groups'][0]['lr']  # 恢复学习率
                self.set_learning_rate(self.learning_rate)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Learning rate: {self.learning_rate}")
            self.current_learning_iteration = loaded_dict["iter"]  # 恢复迭代数
            return loaded_dict["infos"]  # 返回附加信息

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")  # 保存 checkpoint
        torch.save({
            'actor_state_dict': self.actor.state_dict(),  # 保存 actor 参数
            'optimizer_state_dict': self.optimizer.state_dict(),  # 保存优化器参数
            'iter': self.current_learning_iteration,  # 保存当前迭代数
            'infos': infos,  # 保存附加信息
        }, path)
        
    def learn(self):
        if self.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))  # 随机初始化回合长度
        
        obs_dict = self.env.reset_all()  # 重置环境
        for obs_key in obs_dict.keys():
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)  # 转到设备
            
        self._train_mode()  # 切换到训练模式

        num_learning_iterations = self.num_learning_iterations  # 总训练迭代数

        tot_iter = self.current_learning_iteration + num_learning_iterations  # 训练终止迭代数
        
        # for it in track(range(self.current_learning_iteration, tot_iter), description="Learning Iterations"):
        for it in range(self.current_learning_iteration, tot_iter):
            self.start_time = time.time()  # 记录开始时间

            # Jiawei: Need to return obs_dict to update the obs_dict for the next iteration
            # 否则会一直用初始 obs_dict
            obs_dict =self._rollout_step(obs_dict)  # rollout 一轮

            mean_bc_loss = self._training_step()  # 行为克隆损失

            self.stop_time = time.time()  # 记录结束时间
            self.learn_time = self.stop_time - self.start_time  # 计算学习耗时

            # Logging
            log_dict = {
                'it': it,
                'collection_time': self.collection_time,
                'learn_time': self.learn_time,
                'mean_bc_loss': mean_bc_loss,
                'ep_infos': self.ep_infos,
                'rewbuffer': self.rewbuffer,
                'lenbuffer': self.lenbuffer,
                'num_learning_iterations': num_learning_iterations
            }
            self._post_epoch_logging(log_dict)  # 日志记录
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))  # 定期保存模型
            self.ep_infos.clear()  # 清空回合信息
        
        self.current_learning_iteration += num_learning_iterations  # 更新迭代数
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))  # 保存最终模型

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():  # 禁用梯度，提升推理速度
            for i in range(self.num_steps_per_env):
                # Compute the actions and values
                # 计算 agent 和专家的动作
                actions = self.actor.act(obs_dict).detach()  # agent 动作
                action_mean = self.actor.action_mean.detach()  # agent 动作均值
                # action_sigma = self.actor.action_std.detach()
                # actions_log_prob = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
                
                gt_actions = self.gt_actor.act(obs_dict).detach()  # 专家动作
                gt_action_mean = self.gt_actor.action_mean.detach()  # 专家动作均值
                # gt_action_sigma = self.gt_actor.action_std.detach()
                # gt_action_log_prob = self.gt_actor.get_actions_log_prob(gt_actions).detach().unsqueeze(1)
                
                assert len(actions.shape) == 2  # 检查动作 shape
                # assert len(actions_log_prob.shape) == 2
                assert len(action_mean.shape) == 2  # 检查均值 shape
                # assert len(action_sigma.shape) == 2

                policy_state_dict = dict(
                    actions=actions,
                    # actions_log_prob=actions_log_prob,
                    action_mean=action_mean,
                    # action_sigma=action_sigma,
                    gt_actions=gt_actions,
                    # gt_actions_log_prob=gt_action_log_prob,
                    gt_action_mean=gt_action_mean,
                    # gt_action_sigma=gt_action_sigma,
                )  # 组装策略状态字典

                ## Append states to storage
                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])  # 存储观测

                for obs_ in policy_state_dict.keys():
                    self.storage.update_key(obs_, policy_state_dict[obs_])  # 存储策略状态
                    
                actions = policy_state_dict["actions"]  # 获取 agent 动作
                obs_dict, rewards, dones, infos = self.env.step(actions)  # 环境步进
                # critic_obs = privileged_obs if privileged_obs is not None else obs
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)  # 转到设备
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                self.episode_env_tensors.add(infos["to_log"])  # 记录环境日志
                # rewards_stored = rewards.clone().unsqueeze(1)
                # if 'time_outs' in infos:
                #     rewards_stored += self.gamma * policy_state_dict['values'] * infos['time_outs'].unsqueeze(1).to(self.device)
                # assert len(rewards_stored.shape) == 2

                self.storage.increment_step()  # 存储步数加一

                self._process_env_step(rewards, dones, infos)  # 处理环境步进

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])  # 记录回合信息
                    self.cur_reward_sum += rewards  # 累加奖励
                    self.cur_episode_length += 1  # 累加回合长度
                    new_ids = (dones > 0).nonzero(as_tuple=False)  # 找到终止的环境
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())  # 记录奖励
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())  # 记录长度
                    self.cur_reward_sum[new_ids] = 0  # 重置奖励
                    self.cur_episode_length[new_ids] = 0  # 重置长度

            self.stop_time = time.time()  # 记录采集结束时间
            self.collection_time = self.stop_time - self.start_time  # 采集耗时
            self.start_time = self.stop_time  # 更新开始时间
            
            # prepare data for training
            # 为训练准备数据

            # returns, advantages = self._compute_returns(
            #     last_obs_dict=obs_dict,
            #     policy_state_dict=dict(values=self.storage.query_key('values'), 
            #     dones=self.storage.query_key('dones'), 
            #     rewards=self.storage.query_key('rewards'))
            # )
            # self.storage.batch_update_data('returns', returns)
            # self.storage.batch_update_data('advantages', advantages)

        return obs_dict  # 返回最新观测

    def _process_env_step(self, rewards, dones, infos):
        self.actor.reset(dones)  # 根据 dones 重置 actor 状态
 
    def _training_step(self):
        
        mean_bc_loss = 0.0  # 行为克隆损失均值

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)  # mini-batch 生成器

        for obs_dict in generator:
            # Move everything to the device
            for obs_key in obs_dict.keys():
                obs_dict[obs_key] = obs_dict[obs_key].to(self.device)  # 转到设备

            # actions_batch = obs_dict['actions']
            # old_actions_log_prob_batch = obs_dict['actions_log_prob']
            # old_mu_batch = obs_dict['action_mean']
            # old_sigma_batch = obs_dict['action_sigma']
            
            gt_actions_batch = obs_dict['gt_actions']  # 专家动作 batch
            
            self.actor.act(obs_dict,)  # 前向推理
            # actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            mu_batch = self.actor.action_mean  # agent 动作均值
            # sigma_batch = self.actor.action_std
            # entropy_batch = self.actor.entropy
            
            bc_loss = torch.square(mu_batch - gt_actions_batch).mean()  # 行为克隆损失

           
            loss = self.config.bc_loss_coef * bc_loss  # 总损失

            # Gradient step
            self.optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # 梯度裁剪
            self.optimizer.step()  # 优化器步进

            mean_bc_loss += bc_loss.item()  # 累加损失
            # mean_value_loss += value_loss.item()
            # mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches  # 总更新步数
        mean_bc_loss /= num_updates  # 取均值
        self.storage.clear()  # 清空存储

        return mean_bc_loss  # 返回均值

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate  # 设置学习率

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs  # 累加总步数
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']  # 累加总耗时
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']  # 单次迭代耗时

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""  # 记录每回合均值
        mean_std = self.actor.std.mean()  # 动作噪声均值
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))  # 每秒步数

        self.writer.add_scalar('Loss/learning_rate', self.learning_rate, log_dict['it'])  # 记录学习率
        self.writer.add_scalar('Loss/mean_bc_loss', log_dict['mean_bc_loss'], log_dict['it'])  # 记录 BC 损失
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), log_dict['it'])  # 记录噪声
        self.writer.add_scalar('Perf/total_fps', fps, log_dict['it'])  # 记录 FPS
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])  # 记录采集时间
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])  # 记录学习时间
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])  # 记录平均奖励
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])  # 记录平均回合长度
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)  # 记录奖励/时间
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)  # 记录长度/时间

        env_log_dict = self.episode_env_tensors.mean_and_clear()  # 环境统计
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])  # 记录环境统计

        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'BC loss:':>{pad}} {log_dict['mean_bc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")  # 日志字符串
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")  # 追加统计信息
        log_string += f"Logging Directory: {self.log_dir}"  # 日志目录

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass  # 用 rich 实时刷新日志面板

    ##########################################################################################
    # Code for Evaluation
    # 评估相关代码
    ##########################################################################################

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state["actions"])  # 环境步进
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )  # 更新 actor_state
        return actor_state

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.env.reset_all()  # 重置环境
        for obs_key in obs_dict.keys():
            logger.info(f"{obs_key}, {sorted(self.env.config.obs.obs_dict[obs_key])}")  # 打印观测键
        # move to cpu
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()  # 转到 CPU
        return obs_dict

    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()  # 创建评估回调
        self._pre_evaluate_policy()  # 评估前准备
        actor_state = self._create_actor_state()  # 初始化 actor_state
        step = 0
        self.eval_policy = self._get_inference_policy()  # 获取推理策略
        obs_dict = self.env.reset_all()  # 重置环境
        init_actions = torch.zeros(self.env.num_envs, self.num_act, device=self.device)  # 初始化动作
        actor_state.update({"obs": obs_dict, "actions": init_actions})  # 更新 actor_state
        actor_state = self._pre_eval_env_step(actor_state)  # 评估步进前
        while True:
            actor_state["step"] = step  # 步数计数
            actor_state = self._pre_eval_env_step(actor_state)  # 评估步进前
            actor_state = self.env_step(actor_state)  # 环境步进
            actor_state = self._post_eval_env_step(actor_state)  # 评估步进后
            step += 1
        self._post_evaluate_policy()  # 评估后处理

    def _create_actor_state(self):
        return {"done_indices": [], "stop": False}  # 创建 actor_state

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))  # 实例化回调

    def _pre_evaluate_policy(self, reset_env=True):
        self._eval_mode()  # 切换到评估模式
        self.env.set_is_evaluating()  # 设置环境为评估状态
        if reset_env:
            _ = self.env.reset_all()  # 重置环境

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()  # 回调前处理

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()  # 回调后处理

    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"])  # 推理动作
        actor_state.update({"actions": actions})  # 更新 actor_state
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)  # 回调前处理
        return actor_state

    def _post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)  # 回调后处理
        return actor_state

    @property
    def inference_model(self):
        return self.actor  # 返回 actor

    def _get_inference_policy(self, device=None):
        self.actor.eval() # switch to evaluation mode (dropout for example)
        # 切换到评估模式（如禁用 dropout）
        if device is not None:
            self.actor.to(device)  # 转到指定设备
        return self.actor.act_inference  # 返回推理函数