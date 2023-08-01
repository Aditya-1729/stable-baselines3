import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import RL_agent_2
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.controllers import load_controller_config
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir

initialize_config_dir(version_base=None, config_dir="/hpcwork/ru745256/master_thesis/30_6/robosuite/robosuite/main/config")
cfg = compose(config_name="main")

env = RL_agent_2(suite.make(env_name=cfg.env.name,
                **cfg.env.specs,
                task_config = OmegaConf.to_container(cfg.task_config),
                controller_configs=OmegaConf.to_container(cfg.controller)), cfg
)


eval_env = env
# Use deterministic actions for evaluation

# env = gym.make('Pendulum-v1')
# new_logger = configure(["stdout", "csv", "tensorboard"])
model = SAC(env=env, **cfg.algorithm.model)

eval_callback = EvalCallback(eval_env=eval_env, **cfg.algorithm.eval)


# # # model.set_logger(new_logger)
model.learn(**cfg.algorithm.learn, callback=eval_callback)
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

