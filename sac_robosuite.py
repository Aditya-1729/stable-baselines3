import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from robosuite.controllers import load_controller_config

env = GymWrapper(
        suite.make(
            "Wipe",
            robots="Panda",
            controller_configs= load_controller_config(default_controller="OSC_POSE"),  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=False,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

eval_env = env
# Use deterministic actions for evaluation


# env = gym.make('Pendulum-v1')
# new_logger = configure(["stdout", "csv", "tensorboard"])
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_train_eval/",
            learning_starts=3300, batch_size=128, target_update_interval=5,
             train_freq=(1,"step"), tau=0.005, )
# env = model.get_env()[0]

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=10000, 
                             deterministic=True, render=False, n_eval_episodes=5)



# # model.set_logger(new_logger)
model.learn(total_timesteps=10000, log_interval=2, callback=eval_callback)
# model.save("sac_robosuite_2")

# del model # remove to demonstrate saving and loading


# model = SAC.load("sac_robosuite")
# obs,_ = env.reset()
# for i in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, _, _,_, info = env.step(action)
#     env.render()
    # VecEnv resets automatically
    # if done:
#   obs = env.reset()

