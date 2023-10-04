from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, Logger
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, EveryNTimesteps
from stable_baselines3.common.utils import safe_mean
import sys
import time
from hydra import initialize_config_dir, compose
import os

# scriptDir = os.path.dirname(os.path.realpath(__file__))

# initialize_config_dir(version_base=None, \
# config_dir=os.path.abspath('/media/aditya/OS/Users/Aditya/Documents/Uni_Studies/Thesis/master_thesis/1_8/robosuite/stable-baselines3/guide_policy/'))
# cfg = compose(config_name="config")

class HorizonUpdate(BaseCallback):
    def __init__(self, policy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy


    def _on_step(self) -> bool:
        self.policy.update_horizon()
        return True


class Guide_policy:
    def __init__(self):
        # self.env=env
        
        self.position_limits=0.02
        self.dist_th=0.02
        self.indent= 0.01183  

    def predict(self,env):
        self.action= np.zeros(env.action_space.shape)
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        self.action[:12] = env.robots[0].controller.guide_policy_gains
        # self.action[:12]=np.array((7,7,7,7,7,7,200,200,200,200,200,200))
        self.action[12:14] = env.site_pos[:2]-eef_pos[:2]
        self.action[-1] = env.site_pos[-1] - self.indent - eef_pos[-1]
        # delta = eef_pos - env.site_pos
        # dist = np.linalg.norm(delta)
        # print(f"dist:{dist} site: {self.site}")

        return self.action

def get_tbc_policy(ExplorationPolicy: BasePolicy):
    class TBCPolicy(ExplorationPolicy):
        def __init__(
            self,
            *args,
            guide_policy = None,
            max_horizon: int = 0,
            horizons: List[int] = [0],
            strategy: str = "time",
            eval_freq: int = 20,
            n_eval_episodes: int = 20,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.guide_policy = guide_policy
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
            # self.guide_horizon= self.horizons[self.horizon_step]
            return self.horizons[self.horizon_step]
    

        def update_horizon(self) -> None:
            """
            Update the horizon based on the current strategy.
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

        # def predict(
        #     self,
        #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
        #     timesteps: np.ndarray,
        #     state: Optional[Tuple[np.ndarray, ...]] = None,
        #     episode_start: Optional[np.ndarray] = None,
        #     deterministic: bool = False,
            
        # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    
    
        #     # print(f"lambda:{self.lambda_}")
        #     # env.envs[0].env.lambda_ = self.lambda_
        #     action, state_ = super().predict(
        #             observation, state, episode_start, deterministic
        #         )
        #     # action = HybridPolicy(action, self.guide_policy, self.lambda_, env).predict()

        #     return action, state_

    return TBCPolicy


def get_tbc_algorithm(Algorithm: BaseAlgorithm):
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
            # self.guide_policy=Guide_policy(env=self.env)

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


class HybridPolicy:
    def __init__(self,action,guide_policy,lambda_,env):
        self.action=action
        self.env= env.envs[0].env
        self.output_max = self.env.robots[0].controller.output_max
        self.output_min = self.env.robots[0].controller.output_min
        self.input_max = self.env.robots[0].controller.input_max
        self.input_min = self.env.robots[0].controller.input_min
        self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
        self.action_output_transform = (self.output_max + self.output_min) / 2.0
        self.action_input_transform = (self.input_max + self.input_min) / 2.0
        self.guide_policy=guide_policy
        self.lambda_= lambda_
        self.env.lambda_ = lambda_

    def rescale_agent_delta(self) -> np.ndarray:
        action = np.clip(self.action[0][-3:], self.input_min, self.input_max)
        self.transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform
        return self.transformed_action

    def predict (self):
        eef_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id]
        self.action[0][-3:] = self.rescale_agent_delta()
        guide_action = self.guide_policy.predict(self.env)[True]
        # print(guide_action[0][-3:])
        self.final_action = (1-self.lambda_)*guide_action + self.lambda_*self.action
        self.final_action[0][-3:] = eef_pos + np.clip(guide_action[0][-3:] + self.action[0][-3:], a_min=np.ones(3) * (-self.env.position_limits),\
                            a_max=np.ones(3) * (self.env.position_limits))   
        return self.final_action 