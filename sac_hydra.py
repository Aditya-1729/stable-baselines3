import gymnasium as gym
import robosuite as suite
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.wrappers import RL_agent_2
from robosuite.wrappers import Via_points_sweep
from robosuite.wrappers import Via_points_full
from robosuite.wrappers import Via_points
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from wandb.integration.sb3 import WandbCallback
import datetime

@hydra.main(version_base=None, config_path="/hpcwork/thes1499/10_8/robosuite/robosuite/main/config/", config_name="main")
def main(cfg: DictConfig):

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
                    
    if cfg.controller.agent_config==1:
        wrapped_env = Via_points(base_env,cfg)
    if cfg.controller.agent_config==2:
        wrapped_env=Via_points_sweep(base_env,cfg)
    elif cfg.controller.agent_config==3:
        wrapped_env=Via_points_full(base_env,cfg)

    eval_env = wrapped_env

    model = SAC(env=wrapped_env, **cfg.algorithm.model)

    callbacks = [EvalCallback(eval_env=eval_env, **cfg.algorithm.eval)\
                , WandbCallback(verbose=2)]

    model.learn(**cfg.algorithm.learn, callback=callbacks)

if __name__ == "__main__":
    main()


