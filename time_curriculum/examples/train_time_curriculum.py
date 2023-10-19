import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from time_curriculum.src.tbc.tbc import get_tbc_algorithm, Guide_policy
from Callbacks.test import PerformanceLog, Training_info
import robosuite as suite
from robosuite.wrappers import Via_points_full, PolishingGymWrapper, CurriculumWrapper
from stable_baselines3 import SAC
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
import os
from termcolor import colored
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Callable
from stable_baselines3.common.logger import configure



@hydra.main(version_base=None, config_path="/work/thes1499/19_10/robosuite/robosuite/main/config/", config_name="main")
def main(cfg: DictConfig):

    run = wandb.init(
        config=cfg,
        project=cfg.project,
        name=cfg.experiment,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        )    
    env=CurriculumWrapper(suite.make(env_name=cfg.env.name,
                            **cfg.env.specs,
                            task_config=OmegaConf.to_container(
                                cfg.task_config),
                            controller_configs=OmegaConf.to_container(cfg.controller)))    
    # guide_policy = SAC.load("/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/saved_models/4N.zip").policy
    n = cfg.algorithm.curriculum.steps
    max_horizon = env.env.horizon

    # freq = cfg.algorithm.curriculum.complete_handover*cfg.algorithm.learn.total_timesteps/n
    cfg.algorithm.curriculum.freq = int(cfg.algorithm.curriculum.complete_handover*cfg.algorithm.learn.total_timesteps/n)
    
    freq = cfg.algorithm.curriculum.freq#percentage of training used up for handover
    
    print('Complete handover in', colored(f'{int(freq*n)}', 'green'), ' time steps with freq:', colored(f'{int(freq)}', 'yellow'))
    print('# Curriculum steps: ', colored(f'{n}','green'))
    print('Curriculum:', colored(f'{np.arange(max_horizon, -1, -max_horizon // n)}','green'))
    
    model_cfg = OmegaConf.to_container(cfg.algorithm.model)

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

    model = get_tbc_algorithm(SAC)(
        curr_freq=freq,
        env=env,
        # learning_rate=linear_schedule(0.0005),
        **cfg.algorithm.model,
        policy_kwargs=dict(
            guide_policy=Guide_policy,
            max_horizon=max_horizon,
            strategy="time",
            horizons=np.arange(max_horizon, -1, -max_horizon // n,)
        ),
    )
    
    # Set new logger
    tmp_path = cfg.dir
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    callbacks = [PerformanceLog(eval_env=env, **cfg.algorithm.eval,cfg=cfg)]
    
    model.learn(**cfg.algorithm.learn, callback=callbacks)
    

if __name__ == "__main__":
    main()
