import torch  # 导入 PyTorch 主包
from torch import Tensor  # 导入张量类型
from pathlib import Path  # 路径处理

import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from collections import defaultdict  # 字典工具
from multiprocessing import Process, Value  # 多进程工具

import plotly  # 交互式绘图库
import plotly.graph_objects as go  # plotly 图形对象
from plotly.subplots import make_subplots  # 子图工具
from flask import Flask, render_template  # Web 服务与模板
from flask_socketio import SocketIO  # WebSocket 支持
import threading  # 线程工具
import json  # JSON 处理
from flask import send_file  # 文件发送

from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback  # 回调基类
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase  # 机器人环境基类
from humanoidverse.agents.ppo.ppo import PPO  # PPO 算法

# 力估计分析与可视化回调类
class AnalysisPlotForceEstimator(RL_EvalCallback):
    training_loop: PPO  # 训练主循环类型注解
    env: LeggedRobotBase  # 环境类型注解

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)  # 初始化父类
        env: LeggedRobotBase = self.training_loop.env  # 获取环境对象
        self.env = env
        self.policy = self.training_loop._get_inference_policy()  # 获取推理策略
        self.num_envs = self.env.num_envs  # 并行环境数
        self.logger = WebLogger(self.config.sim_dt)  # 初始化 Web 日志工具
        self.reset_buffers()  # 初始化缓存
        self.log_single_robot = self.config.log_single_robot  # 是否只记录单个机器人

    def reset_buffers(self):
        # 初始化观测、动作等缓存
        self.obs_buf = [[] for _ in range(self.num_envs)]  # 观测缓存
        self.critic_obs_buf = [[] for _ in range(self.num_envs)]  # 价值观测缓存
        self.act_buf = [[] for _ in range(self.num_envs)]  # 动作缓存

    def on_pre_evaluate_policy(self):
        # 策略评估前，记录机器人自由度和各项限制
        self.robot_num_dofs = self.env.num_dofs  # 关节自由度数
        self.log_dof_pos_limits = self.env.dof_pos_limits.cpu().numpy()  # 关节位置限制
        self.log_dof_vel_limits = self.env.dof_vel_limits.cpu().numpy()  # 关节速度限制
        self.log_dof_torque_limits = self.env.torque_limits.cpu().numpy()  # 关节力矩限制
        self.logger.set_robot_limits(self.log_dof_pos_limits, self.log_dof_vel_limits, self.log_dof_torque_limits)  # 设置日志限制
        self.logger.set_robot_num_dofs(self.robot_num_dofs)  # 设置自由度数

    def on_post_evaluate_policy(self):
        # 策略评估后钩子（可扩展）
        pass

    def on_pre_eval_env_step(self, actor_state):
        # 环境步进前，记录观测、动作和力估计
        obs: Tensor = actor_state["obs"]["actor_obs"].cpu()  # 策略观测
        critic_obs: Tensor = actor_state["obs"]["critic_obs"].cpu()  # 价值观测
        actions: Tensor = actor_state["actions"].cpu()  # 动作
        left_force_estimated = actor_state["left_hand_force_estimator_output"].cpu()  # 左手力估计
        right_force_estimated = actor_state["right_hand_force_estimator_output"].cpu()  # 右手力估计

        for i in range(self.num_envs):
            self.obs_buf[i].append(obs[i])  # 缓存观测
            self.critic_obs_buf[i].append(critic_obs[i])  # 缓存价值观测
            self.act_buf[i].append(actions[i])  # 缓存动作
    
        rigid_body_pos = self.env._rigid_body_pos_extend  # 刚体位置
        ref_body_pos = self.env.ref_body_pos_extend  # 参考刚体位置
        vr_id = self.env.motion_tracking_id  # 运动跟踪 id

        if self.log_single_robot:
            # 记录单个机器人各项状态
            self.logger.log_states(
                {
                'dof_pos_target': actions[0].cpu().numpy(),  # 目标关节位置
                'dof_pos': self.env.simulator.dof_pos[0].cpu().numpy(),  # 当前关节位置
                'dof_vel': self.env.simulator.dof_vel[0].cpu().numpy(),  # 当前关节速度
                'dof_torque': self.env.torques[0].cpu().numpy(),  # 当前关节力矩
                'base_vel_x': self.env.base_lin_vel[0, 0].item(),  # 基座线速度 x
                'base_vel_y': self.env.base_lin_vel[0, 1].item(),  # 基座线速度 y
                'base_vel_z': self.env.base_lin_vel[0, 2].item(),  # 基座线速度 z
                'base_vel_yaw': self.env.base_ang_vel[0, 2].item(),  # 基座角速度 yaw
                'contact_forces_z': self.env.simulator.contact_forces[0, self.env.feet_indices, 2].cpu().numpy(),  # 足端接触力 z
                'left_hand_force': self.env.apply_force_tensor[0, self.env.left_hand_link_index, :].cpu().numpy(),  # 左手实际力
                'right_hand_force': self.env.apply_force_tensor[0, self.env.right_hand_link_index, :].cpu().numpy(),  # 右手实际力
                'vr_3point_pos': ref_body_pos[0, vr_id, :].cpu().numpy(),  # 参考三点位置
                'vr_3point_pos_target': rigid_body_pos[0, vr_id, :].cpu().numpy(),  # 目标三点位置
                'left_force_estimated': left_force_estimated[0].cpu().numpy(),  # 左手估计力
                'right_force_estimated': right_force_estimated[0].cpu().numpy(),  # 右手估计力
                'feedforward_residual_tau': self.env.feedforward_residual_tau[0].cpu().numpy(),  # 前馈残差力矩
                'feedback_residual_tau': self.env.feedback_residual_tau[0].cpu().numpy(),  # 反馈残差力矩
                }
            )
        else:
            # 记录所有机器人平均状态
            self.logger.log_states(
                {
                    'dof_pos_target': actions.mean(dim=0).cpu().numpy(),  # 平均目标关节位置
                    'dof_pos': self.env.simulator.dof_pos.mean(dim=0).cpu().numpy(),  # 平均关节位置
                    'dof_vel': self.env.simulator.dof_vel.mean(dim=0).cpu().numpy(),  # 平均关节速度
                    'dof_torque': self.env.torques.mean(dim=0).cpu().numpy(),  # 平均关节力矩
                    'base_vel_x': self.env.base_lin_vel[:, 0].mean().item(),
                    'base_vel_y': self.env.base_lin_vel[:, 1].mean().item(),
                    'base_vel_z': self.env.base_lin_vel[:, 2].mean().item(),
                    'base_vel_yaw': self.env.base_ang_vel[:, 2].mean().item(),
                    'contact_forces_z': self.env.simulator.contact_forces[:, self.env.feet_indices, 2].mean(dim=1).cpu().numpy(),
                    'left_hand_force': self.env.apply_force_tensor[:, self.env.left_hand_link_index, :].mean(dim=0).cpu().numpy(),
                    'right_hand_force': self.env.apply_force_tensor[:, self.env.right_hand_link_index, :].mean(dim=0).cpu().numpy(),
                    'vr_3point_pos': ref_body_pos[:, vr_id, :].mean(dim=0).cpu().numpy(),
                    'vr_3point_pos_target': rigid_body_pos[:, vr_id, :].mean(dim=0).cpu().numpy(),
                    'left_force_estimated': left_force_estimated.mean(dim=0).cpu().numpy(),
                    'right_force_estimated': right_force_estimated.mean(dim=0).cpu().numpy,
                    'feedforward_residual_tau': self.env.feedforward_residual_tau.mean(dim=0).cpu().numpy(),
                    'feedback_residual_tau': self.env.feedback_residual_tau.mean(dim=0).cpu().numpy(),
                }
            )
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        step = actor_state["step"]
        if (step + 1) % self.config.plot_update_interval == 0:
            self.logger.plot_states()
        return actor_state

class WebLogger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.thread = None

    def set_robot_limits(self, dof_pos_limits, dof_vel_limits, dof_torque_limits):
        self.log_dof_pos_limits = dof_pos_limits
        self.log_dof_vel_limits = dof_vel_limits
        self.log_dof_torque_limits = dof_torque_limits

    def set_robot_num_dofs(self, num_dofs):
        self.robot_num_dofs = num_dofs

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_server)
            self.thread.start()
        self._update_plot()
    
    def _run_server(self):
        @self.app.route('/')
        def index():
            return send_file('analysis_plot_template.html')

        self.socketio.run(self.app, debug=False, use_reloader=False)
    
    def _update_plot(self):
        log = self.state_log.copy()
        total_time = len(next(iter(log.values()))) * self.dt
        time = np.linspace(0, total_time, len(next(iter(log.values()))))

        for key in log:
            if isinstance(log[key], list):
                log[key] = np.array(log[key])

        BLUE = '#005A9D'
        RED = '#DA2513'
        YELLOW = '#EEDE70'
        BLACK = '#000000'

        force_to_position_scale = 0.05 # force to position scale, for better visualization
        force_to_position_bias = 0.0 # bias to force to position, for better visualization

        num_dofs = self.robot_num_dofs

        def get_subplot_titles():
            titles = [
                'Left hand x',
                'Left hand y',
                'Left hand z',
                'Left hand force',
                'Right hand x',
                'Right hand y',
                'Right hand z',
                'Right hand force',
                'Head x',
                'Head y',
                'Head z',
                'Vertical Contact forces', 
                'Left hand error x',
                'Left hand error y',
                'Left hand error z',
                'Left hand error',
                'Right hand error x',
                'Right hand error y',
                'Right hand error z',
                'Right hand error',
                'Left hand force x',
                'Left hand force y',
                'Left hand force z',
                'Left hand force',
                'Right hand force x',
                'Right hand force y',
                'Right hand force z',
                'Right hand force',
                'Base velocity x', 
                'Base velocity y', 
                'Base velocity yaw', 
                'Base velocity z',
            ]
            for i in range(num_dofs):
                titles.extend([f'DOF {i} Position', f'DOF {i} Velocity', f'DOF {i} Torque', f'DOF {i} Torque/Velocity'])
            return titles

        # Calculate number of rows needed
        num_rows = 8 + num_dofs

        fig = make_subplots(rows=num_rows, cols=4, subplot_titles=get_subplot_titles())

        def add_trace(x, y, color, row, col, name=None, show_legend=False):
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color), name=name, showlegend=show_legend), row=row, col=col)

        add_trace(time, log["vr_3point_pos"][:, 0, 0], BLUE, 1, 1, "Left Hand Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 0], RED, 1, 1, "Left Hand Position Target x")
        # add_trace(time, log["left_hand_force"][:, 0] * force_to_position_scale + force_to_position_bias, RED, 1, 1, "Left Hand Force x")

        add_trace(time, log["vr_3point_pos"][:, 0, 1], BLUE, 1, 2, "Left Hand Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 1], RED, 1, 2, "Left Hand Position Target y")
        # add_trace(time, log["left_hand_force"][:, 1] * force_to_position_scale + force_to_position_bias, RED, 1, 2, "Left Hand Force y")

        add_trace(time, log["vr_3point_pos"][:, 0, 2], BLUE, 1, 3, "Left Hand Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 2], RED, 1, 3, "Left Hand Position Target z")
        # add_trace(time, log["left_hand_force"][:, 2] * force_to_position_scale + force_to_position_bias, RED, 1, 3, "Left Hand Force z")

        add_trace(time, np.linalg.norm(log["left_hand_force"], axis=1), RED, 1, 4, "Left Hand Force")

        add_trace(time, log["vr_3point_pos"][:, 1, 0], BLUE, 2, 1, "Right Hand Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 0], RED, 2, 1, "Right Hand Position Target x")
        # add_trace(time, log["right_hand_force"][:, 0] * force_to_position_scale + force_to_position_bias, RED, 2, 1, "Right Hand Force x")

        add_trace(time, log["vr_3point_pos"][:, 1, 1], BLUE, 2, 2, "Right Hand Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 1], RED, 2, 2, "Right Hand Position Target y")
        # add_trace(time, log["right_hand_force"][:, 1] * force_to_position_scale + force_to_position_bias, RED, 2, 2, "Right Hand Force y")

        add_trace(time, log["vr_3point_pos"][:, 1, 2], BLUE, 2, 3, "Right Hand Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 2], RED, 2, 3, "Right Hand Position Target z")
        # add_trace(time, log["right_hand_force"][:, 2] * force_to_position_scale + force_to_position_bias, RED, 2, 3, "Right Hand Force z")

        add_trace(time, np.linalg.norm(log["right_hand_force"], axis=1), RED, 2, 4, "Right Hand Force")

        add_trace(time, log["vr_3point_pos"][:, 2, 0], BLUE, 3, 1, "Head Position x")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 0], BLACK, 3, 1, "Head Position Target x")

        add_trace(time, log["vr_3point_pos"][:, 2, 1], BLUE, 3, 2, "Head Position y")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 1], BLACK, 3, 2, "Head Position Target y")

        add_trace(time, log["vr_3point_pos"][:, 2, 2], BLUE, 3, 3, "Head Position z")
        add_trace(time, log["vr_3point_pos_target"][:, 2, 2], BLACK, 3, 3, "Head Position Target z")

        # Vertical Contact forces
        forces = log["contact_forces_z"]
        for i in range(forces[0].shape[0]):
            add_trace(time, [force[i] for force in forces], BLUE, 3, 4, f"Force {i}")

        add_trace(time, log["left_hand_force"][:, 2], BLUE, 3, 4, "Left Hand Force z")
        add_trace(time, log["right_hand_force"][:, 2], RED, 3, 4, "Right Hand Force z")

        # error

        add_trace(time, log["vr_3point_pos_target"][:, 0, 0] - log["vr_3point_pos"][:, 0, 0], RED, 4, 1, "Left Hand Error x")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 1] - log["vr_3point_pos"][:, 0, 1], RED, 4, 2, "Left Hand Error y")
        add_trace(time, log["vr_3point_pos_target"][:, 0, 2] - log["vr_3point_pos"][:, 0, 2], RED, 4, 3, "Left Hand Error z")
        add_trace(time, np.linalg.norm(log["vr_3point_pos_target"][:, 0] - log["vr_3point_pos"][:, 0], axis=1), RED, 4, 4, "Left Hand Error")

        add_trace(time, log["vr_3point_pos_target"][:, 1, 0] - log["vr_3point_pos"][:, 1, 0], RED, 5, 1, "Right Hand Error x")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 1] - log["vr_3point_pos"][:, 1, 1], RED, 5, 2, "Right Hand Error y")
        add_trace(time, log["vr_3point_pos_target"][:, 1, 2] - log["vr_3point_pos"][:, 1, 2], RED, 5, 3, "Right Hand Error z")
        add_trace(time, np.linalg.norm(log["vr_3point_pos_target"][:, 1] - log["vr_3point_pos"][:, 1], axis=1), RED, 5, 4, "Right Hand Error")

        add_trace(time, log["left_hand_force"][:, 0], RED, 6, 1, "Left Hand Force x")
        add_trace(time, log["left_force_estimated"][:, 0], YELLOW, 6, 1, "Estimated Left Hand Force x")

        add_trace(time, log["left_hand_force"][:, 1], RED, 6, 2, "Left Hand Force y")
        add_trace(time, log["left_force_estimated"][:, 1], YELLOW, 6, 2, "Estimated Left Hand Force y")

        add_trace(time, log["left_hand_force"][:, 2], RED, 6, 3, "Left Hand Force z")
        add_trace(time, log["left_force_estimated"][:, 2], YELLOW, 6, 3, "Estimated Left Hand Force z")

        add_trace(time, np.linalg.norm(log["left_hand_force"], axis=1), RED, 6, 4, "Left Hand Force")
        add_trace(time, np.linalg.norm(log["left_force_estimated"], axis=1), YELLOW, 6, 4, "Estimated Left Hand Force")

        add_trace(time, log["right_hand_force"][:, 0], RED, 7, 1, "Right Hand Force x")
        add_trace(time, log["right_force_estimated"][:, 0], YELLOW, 7, 1, "Estimated Right Hand Force x")
        add_trace(time, log["right_hand_force"][:, 1], RED, 7, 2, "Right Hand Force y")
        add_trace(time, log["right_force_estimated"][:, 1], YELLOW, 7, 2, "Estimated Right Hand Force y")
        add_trace(time, log["right_hand_force"][:, 2], RED, 7, 3, "Right Hand Force z")
        add_trace(time, log["right_force_estimated"][:, 2], YELLOW, 7, 3, "Estimated Right Hand Force z")
        add_trace(time, np.linalg.norm(log["right_hand_force"], axis=1), RED, 7, 4, "Right Hand Force")
        add_trace(time, np.linalg.norm(log["right_force_estimated"], axis=1), YELLOW, 7, 4, "Estimated Right Hand Force")
        
        def add_limit_lines(row, col, lower, upper, color=YELLOW):
            fig.add_shape(type="rect", x0=time[0], x1=time[-1], y0=lower, y1=upper,
                        fillcolor=color, line=dict(width=0), layer='below', row=row, col=col)

        # DOF Positions, Velocities, and Torques
        for i in range(num_dofs):
            row = i + 8  # Start from the third row

            # Position
            add_trace(time, [pos[i] for pos in log["dof_pos"]], BLUE, row, 1, f"DOF {i} pos")
            add_trace(time, [pos[i] for pos in log["dof_pos_target"]], RED, row, 1, f"DOF {i} pos target")
            add_limit_lines(row, 1, self.log_dof_pos_limits[i, 0], self.log_dof_pos_limits[i, 1])
            # Velocity
            add_trace(time, [vel[i] for vel in log["dof_vel"]], BLUE, row, 2, f"DOF {i} vel")
            add_limit_lines(row, 2, -self.log_dof_vel_limits[i], self.log_dof_vel_limits[i])
            # Torque
            add_trace(time, [torque[i] for torque in log["dof_torque"]], BLUE, row, 3, f"DOF {i} torque")
            add_limit_lines(row, 3, -self.log_dof_torque_limits[i], self.log_dof_torque_limits[i])

            add_trace(time, [torque[i] for torque in log["feedforward_residual_tau"]], RED, row, 3, f"Feedforward Residual Torque")
            add_trace(time, [torque[i] for torque in log["feedback_residual_tau"]], BLACK, row, 3, f"Feedback Residual Torque")
            
            # Torque/Velocity curve
            fig.add_trace(go.Scatter(
                x=[vel[i] for vel in log["dof_vel"]], 
                y=[torque[i] for torque in log["dof_torque"]], 
                mode='markers', 
                marker=dict(color=BLUE, size=2), 
                showlegend=False,
                name=f"DOF {i} Torque/Velocity"
            ), row=row, col=4)

            # Add velocity limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=-self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

            # Add torque limits
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=-self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=-self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)
            fig.add_shape(type="line", x0=-self.log_dof_vel_limits[i], y0=self.log_dof_torque_limits[i], 
                        x1=self.log_dof_vel_limits[i], y1=self.log_dof_torque_limits[i],
                        line=dict(color=YELLOW, width=2), row=row, col=4)

        fig.update_layout(height=300*num_rows, width=1500, title_text="Robot State Plots", showlegend=True)
        
        # Update x and y axis labels
        for i in range(num_rows):
            for j in range(3):
                fig.update_xaxes(title_text="time [s]", row=i+1, col=j+1)
            fig.update_xaxes(title_text="", row=i+1, col=4)

        
        
        fig.update_yaxes(title_text="Left Hand Position x [m]", row=1, col=1)
        fig.update_yaxes(title_text="Left Hand Position y [m]", row=1, col=2)
        fig.update_yaxes(title_text="Left Hand Position z [m]", row=1, col=3)
        fig.update_yaxes(title_text="Left Hand Force [N]", row=1, col=4)
        fig.update_yaxes(title_text="Right Hand Position x [m]", row=2, col=1)
        fig.update_yaxes(title_text="Right Hand Position y [m]", row=2, col=2)
        fig.update_yaxes(title_text="Right Hand Position z [m]", row=2, col=3)
        fig.update_yaxes(title_text="Right Hand Force [N]", row=2, col=4)

        fig.update_yaxes(title_text="Head Position x [m]", row=3, col=1)
        fig.update_yaxes(title_text="Head Position y [m]", row=3, col=2)
        fig.update_yaxes(title_text="Head Position z [m]", row=3, col=3)

        fig.update_yaxes(title_text="Vertical Contact Forces [N]", row=3, col=4)

        for i in range(4, num_rows + 1):
            fig.update_yaxes(title_text="Position [rad]", row=i, col=1)
            fig.update_yaxes(title_text="Velocity [rad/s]", row=i, col=2)
            fig.update_yaxes(title_text="Torque [Nm]", row=i, col=3)
            fig.update_yaxes(title_text="Torque/Velocity", row=i, col=4)

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        self.socketio.emit('update_plots', plot_json)

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.thread:
            self.socketio.stop()
            self.thread.join()