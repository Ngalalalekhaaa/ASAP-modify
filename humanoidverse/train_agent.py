import os  # 导入操作系统相关模块
import sys  # 导入系统相关模块
from pathlib import Path  # 导入Path用于路径操作

import hydra  # 导入hydra用于配置管理
from hydra.core.hydra_config import HydraConfig  # hydra配置相关
from hydra.core.config_store import ConfigStore  # hydra配置存储
from hydra.utils import instantiate  # hydra用于实例化对象
from omegaconf import OmegaConf  # 配置对象

import logging  # 标准日志模块
from loguru import logger  # 第三方日志库loguru




from utils.config_utils import *  # noqa: E402, F403  # 导入自定义配置工具


@hydra.main(config_path="config", config_name="base", version_base="1.1")  # hydra主入口，指定配置路径和名称
def main(config: OmegaConf):  # 主函数，接收OmegaConf配置
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator['_target_'].split('.')[-1]  # 获取仿真器类型
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacSim':  # 如果是IsaacSim仿真器
        from omni.isaac.lab.app import AppLauncher  # 导入IsaacSim启动器
        import argparse  # 导入命令行参数解析模块
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")  # 创建参数解析器
        AppLauncher.add_app_launcher_args(parser)  # 添加IsaacSim相关参数
        
        args_cli, hydra_args = parser.parse_known_args()  # 解析已知和未知参数
        sys.argv = [sys.argv[0]] + hydra_args  # 重置sys.argv以兼容hydra
        args_cli.num_envs = config.num_envs  # 设置环境数量
        args_cli.seed = config.seed  # 设置随机种子
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing  # 设置环境间距
        args_cli.output_dir = config.output_dir  # 设置输出目录
        args_cli.headless = config.headless  # 是否无头模式
        
        app_launcher = AppLauncher(args_cli)  # 创建AppLauncher实例
        simulation_app = app_launcher.app  # 获取仿真应用对象
        
        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':  # 如果是IsaacGym仿真器
        import isaacgym  # noqa: F401  # 导入isaacgym
        
    if simulator_type == 'Genesis' or simulator_type == 'genesis':
        import genesis as gs
        gs.init(backend=gs.gpu) # 初始化 Genesis GPU 后端



    # have to import torch after isaacgym
    import torch  # noqa: E402  # 导入torch
    from utils.common import seeding  # 导入自定义随机种子函数
    import wandb  # 导入wandb用于实验追踪
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402  # 导入基础任务类
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402  # 导入基础算法类
    from humanoidverse.utils.helpers import pre_process_config  # 导入配置预处理函数
    from humanoidverse.utils.logging import HydraLoggerBridge  # 导入Hydra日志桥接
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")  # hydra日志文件路径：由base.hydra.run.dir的值确定
    logger.remove()  # 移除默认日志
    logger.add(hydra_log_path, level="DEBUG")  # 添加文件日志，级别DEBUG

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()  # 获取控制台日志级别
    logger.add(sys.stdout, level=console_log_level, colorize=True)  # 控制台输出日志

    logging.basicConfig(level=logging.DEBUG)  # 标准日志配置
    logging.getLogger().addHandler(HydraLoggerBridge())  # 添加Hydra日志桥接

    unresolved_conf = OmegaConf.to_container(config, resolve=False)  # 转为dict，保持未解析状态
    os.chdir(hydra.utils.get_original_cwd())  # 切换到原始工作目录

    if config.use_wandb:  # 如果使用wandb
        project_name = f"{config.project_name}"  # wandb项目名
        run_name = f"{config.timestamp}_{config.experiment_name}_{config.log_task_name}_{config.robot.asset.robot_type}"  # wandb运行名
        wandb_dir = Path(config.wandb.wandb_dir)  # wandb日志目录
        wandb_dir.mkdir(exist_ok=True, parents=True)  # 创建wandb目录
        logger.info(f"Saving wandb logs to {wandb_dir}")  # 记录wandb日志保存路径
        wandb.init(project=project_name, 
            entity=config.wandb.wandb_entity,  # wandb实体
            name=run_name,  # wandb运行名
            sync_tensorboard=True,  # 同步tensorboard
            config=unresolved_conf,  # 配置
            dir=wandb_dir)  # 日志目录
    
    if hasattr(config, 'device'):  # 判断配置中是否有device
        if config.device is not None:
            device = config.device  # 使用指定device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 自动选择cuda或cpu
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 默认选择cuda或cpu
    
    pre_process_config(config)  # 配置预处理

    # torch.set_float32_matmul_precision("medium")

    # fabric: Fabric = instantiate(config.fabric)
    # fabric.launch()

    # if config.seed is not None:
    #     rank = fabric.global_rank
    #     if rank is None:
    #         rank = 0
    #     fabric.seed_everything(config.seed + rank)
    #     seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)
    config.env.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")  # 渲染保存目录
    env: BaseEnv = instantiate(config=config.env, device=device)  # 实例化环境


    experiment_save_dir = Path(config.experiment_dir)  # 实验保存目录
    experiment_save_dir.mkdir(exist_ok=True, parents=True)  # 创建实验目录

    logger.info(f"Saving config file to {experiment_save_dir}")  # 记录配置保存路径
    with open(experiment_save_dir / "config.yaml", "w") as file:  # 保存配置到yaml
        OmegaConf.save(unresolved_conf, file)  # 保存配置

    algo: BaseAlgo = instantiate(device=device, env=env, config=config.algo, log_dir=experiment_save_dir)  # 实例化算法
    algo.setup()  # 算法初始化
    # import ipdb;    ipdb.set_trace()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)  # 加载模型检查点

    # handle saving config
    algo.learn()  # 开始训练

    if simulator_type == 'IsaacSim':
        simulation_app.close()  # 关闭仿真应用

if __name__ == "__main__":  # 脚本入口
    main()  # 调用主函数
