from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, EveryNTimesteps
from stable_baselines3.common.utils import safe_mean
from Callbacks.test import PerformanceLog, Training_info
import sys
import time
from hydra import initialize_config_dir, compose
import os

class HorizonUpdate(BaseCallback):
    """
    This class updates the horizon of the policy on every step.

    :param policy: The policy to update.
    """
    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy


    def _on_step(self) -> bool:
        self.policy.update_horizon()
        return True

def get_tbc_policy(ExplorationPolicy: BasePolicy):
    """
    This function extends a policy class to include the Time Based Curriculum.
    :param ExplorationPolicy: The policy to be extended.
    :return: A new policy class that includes the Time Based Curriculum.
    """

    class TBCPolicy(ExplorationPolicy):
        def __init__(
            self,
            *args,
            max_horizon: int = 0,
            horizons: List[int] = [0],
            strategy: str = "time",
            eval_freq: int = 20,
            n_eval_episodes: int = 20,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            assert strategy in ["curriculum", "time"], f"strategy: '{strategy}' must be 'curriculum' or 'random'"
            self.strategy = strategy
            self.horizon_step= 0
            self.lambda_= 0
            self.guide_horizon = max_horizon
            self.max_horizon = max_horizon
            self.horizons = horizons
            self.eval_freq = eval_freq
            # self.env = env 
        @property
        def horizon(self):
            return self.horizons[self.horizon_step]
    

        def update_horizon(self) -> None:
            """
            Update the horizon based on the current strategy.
            horizon_step is the index of the current horizon in the list of horizons.
            guide_horizon is the horizon 'rolled in' by the guide policy .
            """
            if self.strategy == "time":
                self.horizon_step += 1
                self.horizon_step = min(self.horizon_step, len(self.horizons) - 1)
                self.guide_horizon = self.horizons[self.horizon_step]
                print(f"horizons:{self.horizons}")
                print(f"horizon_step:{self.horizon_step}")
                print(f"guide_horizon:{self.guide_horizon}")
                # print(f"guide_horizon_prop:{self.horizon()}")
                self.lambda_ = np.clip((self.max_horizon-self.guide_horizon)/self.max_horizon, a_max=1, a_min=0)

    return TBCPolicy


def get_tbc_algorithm(Algorithm: BaseAlgorithm):
    """
    :param Algorithm: The algorithm to be extended.
    :return: A new algorithm class that includes the TBC curriculum.
    """
    class TBCAlgorithm(Algorithm):
        def __init__(self, curr_freq, policy, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_tbc_policy(policy)
            self.curr_freq=curr_freq
            kwargs["learning_starts"] = 0
            super().__init__(policy, *args, **kwargs)

        def _init_callback(
            self,
            callback: MaybeCallback,
            progress_bar: bool = False,
        ) -> BaseCallback:
            """
            :param callback: Callback(s) called at every step with state of the algorithm.
            :param progress_bar: Display a progress bar using tqdm and rich.
            :return: A hybrid callback calling `callback` and performing evaluation.
            """
            callback = super()._init_callback(callback, progress_bar)
            curriculum_callback = EveryNTimesteps(n_steps=self.curr_freq,
                callback=HorizonUpdate(self.policy
                    ),
            )
            callback = CallbackList(
                [
                    callback,
                    Training_info(),
                    curriculum_callback,
                ]
            )
            callback.init_callback(self)
            return callback

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).
            Stores environment lambda_ (used later in the wrapper) from the policy object.
            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            action, state = self.policy.predict(observation, deterministic)
            self.env.envs[0].env.lambda_ = self.policy.lambda_
            return action, state

        def _dump_logs(self):
            """
            Write log.
            """
            time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
            fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
            if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                self.logger.record("rollout/guide_horizon", self.policy.guide_horizon)
                self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("time/fps", fps)
            self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            if self.use_sde:
                self.logger.record("train/std", (self.actor.get_std()).mean().item())

            if len(self.ep_success_buffer) > 0:
                self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
            # Pass the number of timesteps for tensorboard
            self.logger.dump(step=self.num_timesteps)
            
            
    return TBCAlgorithm
