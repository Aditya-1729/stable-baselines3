import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback

def main():
    env = gym.make("PointMaze_UMaze-v3", continuing_task=False, max_episode_steps=100)
    model = TD3(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/pointmaze_guide"
    )
    model.learn(
        total_timesteps=1000,
        log_interval=100,
        # progress_bar=True,
        callback=EvalCallback(
            env,
            verbose=2,
            eval_freq=300,
            n_eval_episodes=2,
            best_model_save_path="./examples/models/pointmaze_guide_TD3"
        ),
    )


if __name__ == "__main__":
    main()
