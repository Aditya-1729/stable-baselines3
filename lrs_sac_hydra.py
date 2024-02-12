import gymnasium as gym
import robosuite as suite
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import Via_points_sweep
from robosuite.wrappers import Via_points_full
from robosuite.wrappers import Via_points
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from Callbacks.test import PerformanceLog, Training_info
from stable_baselines3.common.logger import configure
from typing import Callable
from wandb.integration.sb3 import WandbCallback
import datetime

@hydra.main(version_base=None, config_path="/work/thes1499/DR_19_10/robosuite/robosuite/main/config/", config_name="main")
def main(cfg: DictConfig):
    if cfg.use_wandb:
        run = wandb.init(
            config=cfg,
            project=cfg.project,
            name=cfg.experiment,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )

    base_env = suite.make(env_name=cfg.env.name,
                                **cfg.env.specs,
                                task_config=OmegaConf.to_container(
                                    cfg.task_config),
                                controller_configs=OmegaConf.to_container(cfg.controller))
                    
    if cfg.controller.agent_config==0:
        wrapped_env = GymWrapper(base_env)
        # cfg.controller.control_delta=True    
    if cfg.controller.agent_config==1:
        wrapped_env = Via_points(base_env,cfg)
    if cfg.controller.agent_config==2:
        wrapped_env=Via_points_sweep(base_env,cfg)
    elif cfg.controller.agent_config==3:
        wrapped_env=Via_points_full(base_env,cfg)

    eval_env = wrapped_env

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func
    
    model = SAC(env=wrapped_env, learning_rate=linear_schedule(0.0005), **cfg.algorithm.model)

    
    # set up logger
    tmp_path = cfg.dir
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    callbacks = [PerformanceLog(eval_env=eval_env, **cfg.algorithm.eval, cfg=cfg)\
                , Training_info()]

    model.learn(**cfg.algorithm.learn, callback=callbacks)

if __name__ == "__main__":
    main()


