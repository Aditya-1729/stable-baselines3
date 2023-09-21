import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import robosuite as suite
from robosuite.wrappers import PolishingGymWrapper   
from Residual_RL.src.policies import ResidualSAC
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from Callbacks.test import PerformanceLog
import wandb


@hydra.main(version_base=None, config_path="/hpcwork/thes1499/10_8/robosuite/robosuite/main/config/", config_name="main")
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

    env=PolishingGymWrapper(suite.make(env_name=cfg.env.name,
                            **cfg.env.specs,
                            task_config=OmegaConf.to_container(
                                cfg.task_config),
                            controller_configs=OmegaConf.to_container(cfg.controller)))    



    model = ResidualSAC(env=env,**cfg.algorithm.model)

    callbacks = [PerformanceLog(eval_env=env, **cfg.algorithm.eval, cfg=cfg)]
    
    model.learn(**cfg.algorithm.learn, callback=callbacks)
    
if __name__ == "__main__":
    main()
