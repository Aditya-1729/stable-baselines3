from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, EventCallback
from hydra import compose, initialize
import wandb
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from fractions import Fraction
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.logger import Logger

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

from stable_baselines3.common import base_class  # pytype: disable=pyi-error
# from stable_baselines3.common.evaluation import eval_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

class BestWandb(EvalCallback):
    def __init__(self, cfg, use_wandb=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb=use_wandb
        self.cfg=cfg
    def _on_step(self):
        print('wandb_best_eval is on')
        eval_policy(
            self.use_wandb,
            self.cfg,
            self.model,
            self.eval_env,
            n_eval_episodes=1,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )


def eval_policy(
    use_wandb: bool,
    cfg: DictConfig,
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True) :
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if use_wandb:
        run = wandb.init(
        # Set the project where this run will be logged
        project=cfg.project,
        name=cfg.experiment,
        config=cfg) 
        config = wandb.config

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_force = []
    episode_deviation= []
    episode_via_points=[]
    Return = 0
    
    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                Return+= reward
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                episode_force.append(info['force'])
                episode_deviation.append(info['deviation'])
                episode_via_points.append(info['nwipedmarkers'])

                if callback is not None:
                    callback(locals(), globals())
                if use_wandb:
                    metrics = { 'Wipe/reward': reward,
                                    'Wipe/Return': Return,
                                    'Wipe/force': info['force'],
                                    # 'Wipe/unadjusted_force': env.envs[0].robots[0].ee_force,
                                    "Wipe/task_complete_reward": env.envs[0].task_completion_r,
                                    "Wipe/excess_force_penalty": env.envs[0].force_penalty,
                                    "Wipe/low_force_penalty":env.envs[0].low_force_penalty,
                                    "Wipe/distance_penalty":-env.envs[0].total_dist_reward,
                                    "Wipe/wipe_contact_reward": env.envs[0].wipe_contact_r,
                                    "Wipe/unnormalized_reward": env.envs[0].un_normalized_reward,
                                    "Wipe/unit_wiped_reward": env.envs[0].unit_wipe,
                                    # "Wipe/time_ratio":t_contact/t,
                                    "Wipe/force_in_window_penalty":env.envs[0].force_in_window_penalty,
                                    'Wipe/ncontacts': env.envs[0].sim.data.ncon,
                                    'Wipe/del_x': info['deviation'],
                                    'Wipe/vel_y': env.envs[0].robots[0]._hand_vel[1],
                                    # 'Wipe/del_z': delta[2],
                                    'Wipe/reward_wop': env.envs[0].reward_wop,
                                    }
                    
                    wandb.log({**metrics})
                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()
    
    return episode_rewards, episode_lengths, episode_force, episode_deviation, episode_via_points


class PerformanceLog(EvalCallback):
    def __init__(self, cfg, use_wandb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb
        self.cfg=cfg
    
    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_forces,episode_deviations, episode_wiped= eval_policy(
                self.use_wandb,
                self.cfg,
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)


                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            mean_ep_force, std_ep_force, max_ep_force, min_ep_force = np.mean(episode_forces), np.std(episode_forces) , np.max(episode_forces), np.min(episode_forces)
            mean_ep_force, std_ep_force, max_ep_force, min_ep_force = np.mean(episode_forces), np.std(episode_forces) , np.max(episode_forces), np.min(episode_forces)
            mean_ep_deviation, std_ep_deviation = np.mean(episode_deviations), np.std(episode_deviations)
            ep_last_via_point,ep_total_via_points = episode_wiped[-1], max(episode_wiped) 
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_ep_force", mean_ep_force)
            self.logger.record("eval/std_ep_force", std_ep_force)
            self.logger.record("eval/max_ep_force", max_ep_force)
            self.logger.record("eval/min_ep_force", min_ep_force)
            self.logger.record("eval/max_ep_force", max_ep_force)
            self.logger.record("eval/min_ep_force", min_ep_force)
            self.logger.record("eval/mean_ep_x_dev", mean_ep_deviation)
            self.logger.record("eval/std_ep_x_dev", std_ep_deviation)
            self.logger.record("eval/via_points_wiped", ep_total_via_points)



            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class Training_info(BaseCallback):
    def __init__(self,verbose=0):
        self.collisions=0
        self.f_excess=0
        self.q_limits=0
        super().__init__(verbose)
    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        # print(infos)
        self.collisions += infos["colls"]
        self.f_excess += infos["f_excess"]
        self.q_limits += infos["lims"]
        # self.q_limits += 1  
        self.logger.record("train/collisions", self.collisions)
        self.logger.record("train/f_excess", self.f_excess)
        self.logger.record("train/joint_limits", self.q_limits)
        return True

