import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import RL_agent_2
from robosuite.wrappers import Via_points_sweep
from robosuite.wrappers import Via_points_full
from robosuite.wrappers import Via_points
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.controllers import load_controller_config
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from hydra import compose, initialize_config_dir
from wandb.integration.sb3 import WandbCallback
import datetime

initialize_config_dir(
    version_base=None, config_dir="/hpcwork/thes1499/10_8/robosuite/robosuite/main/config")
cfg = compose(config_name="main")

run = wandb.init(
    project="sb3",
    name="1500_sweep",
    config=cfg,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
)

env = Via_points_full(suite.make(env_name=cfg.env.name,
                            **cfg.env.specs,
                            task_config=OmegaConf.to_container(
                                cfg.task_config),
                            controller_configs=OmegaConf.to_container(cfg.controller))
                 )


eval_env = env
# Use deterministic actions for evaluation

# env = gym.make('Pendulum-v1')
# new_logger = configure(["stdout", "csv", "tensorboard"])
model = SAC(env=env, **cfg.algorithm.model)

# eval_callback = EvalCallback(eval_env=eval_env, **cfg.algorithm.eval)
# wandb_callback = WandbCallback(verbose=2,)
callbacks = [EvalCallback(eval_env=eval_env, best_model_save_path = f'./logs/hpc_1500/complete/{datetime.datetime.now().strftime("%d_%m/%H%M%S")}',\
             **cfg.algorithm.eval), WandbCallback(verbose=2) ]
# # # model.set_logger(new_logger)

model.learn(**cfg.algorithm.learn, callback=callbacks)

# model.save("sac_robosuite_2")

# del model # remove to demonstrate saving and loading


# model = SAC.load("/hpcwork/ru745256/master_thesis/30_6/robosuite/logs/gpu/best_model.zip")
# obs,_ = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, _, _,_, info = env.step(action)
#     env.render()
# VecEnv resets automatically
# if done:
#   obs = env.reset()
