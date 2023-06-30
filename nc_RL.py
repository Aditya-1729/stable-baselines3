import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import RL_agent_2
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.controllers import load_controller_config
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.integration.sb3 import WandbCallback

initialize_config_dir(version_base=None, config_dir="/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/git_clean/robosuite/robosuite/main/config")
cfg = compose(config_name="main")

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
# run = wandb.init(
#     project="sb3",
#     config=cfg,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     monitor_gym=False,  # auto-upload the videos of agents playing the game
#     save_code=False,  # optional
# )


env = RL_agent_2(
    suite.make(env_name=cfg.env.name,
                **cfg.env.specs,
                task_config = OmegaConf.to_container(cfg.task_config),
                controller_configs=OmegaConf.to_container(cfg.controller))
                ,cfg
)

eval_env = env
# Use deterministic actions for evaluation


# env = gym.make('Pendulum-v1')
# new_logger = configure(["stdout", "csv", "tensorboard"])
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_train_eval/",
            learning_starts=0, batch_size=128, target_update_interval=5,
             train_freq=(1,"step"), tau=0.005, )
# env = model.get_env()[0]

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10, 
                             deterministic=True, render=False, n_eval_episodes=1)



# # model.set_logger(new_logger)
model.learn(total_timesteps=1000, log_interval=10)#, callback=eval_callback)
# model.save("sac_robosuite_2")

# del model # remove to demonstrate saving and loading


# model = SAC.load("sac_robosuite")
# obs,_ = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, _, _,_, info = env.step(action)
    # env.render()
    # VecEnv resets automatically
    # if done:
#   obs = env.reset()

