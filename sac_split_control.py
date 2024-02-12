import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import SplitWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from robosuite.controllers import load_controller_config
from omegaconf import DictConfig, OmegaConf
from Callbacks.test import PerformanceLog, Training_info
import wandb
import hydra



@hydra.main(version_base=None, config_path="/work/thes1499/7_10/10_8/robosuite/robosuite/main/config/", config_name="main")
def main(cfg: DictConfig):

    run = wandb.init(
        config=cfg,
        project=cfg.project,
        name=cfg.experiment,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        ) 

    env = SplitWrapper(suite.make(env_name=cfg.env.name,
                                **cfg.env.specs,
                                task_config=OmegaConf.to_container(
                                    cfg.task_config),
                                controller_configs=OmegaConf.to_container(cfg.controller)), cfg
                 )


    model = SAC(env=env, **cfg.algorithm.model)

    tmp_path = cfg.dir
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    model.set_logger(new_logger)
    callbacks = [PerformanceLog(eval_env=env, **cfg.algorithm.eval, cfg=cfg), Training_info()]
    
    model.learn(**cfg.algorithm.learn, callback=callbacks)
    
if __name__ == "__main__":
    main()
