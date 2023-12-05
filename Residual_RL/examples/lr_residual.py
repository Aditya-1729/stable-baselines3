import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import robosuite as suite
from robosuite.wrappers import ResidualWrapper   
from Residual_RL.src.policies import ResidualSAC
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from Callbacks.test import PerformanceLog, Training_info
from Callbacks.test import PerformanceLog, Training_info
import wandb
from stable_baselines3.common.logger import configure
from typing import Callable

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

    env=ResidualWrapper(suite.make(env_name=cfg.env.name,
                            **cfg.env.specs,
                            task_config=OmegaConf.to_container(
                                cfg.task_config),
                            controller_configs=OmegaConf.to_container(cfg.controller)))    

    model = ResidualSAC(env=env,learning_rate=linear_schedule(0.001),**cfg.algorithm.model)    
    
    # Set new logger
    tmp_path = cfg.dir
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    model.set_logger(new_logger)
    callbacks = [PerformanceLog(eval_env=env, **cfg.algorithm.eval, cfg=cfg), Training_info()]
    
    model.learn(**cfg.algorithm.learn, callback=callbacks)
    
if __name__ == "__main__":
    main()
