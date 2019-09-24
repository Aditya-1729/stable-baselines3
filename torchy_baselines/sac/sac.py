import time

import torch as th
import torch.nn.functional as F
import numpy as np

from torchy_baselines.common.base_class import BaseRLModel
from torchy_baselines.common.buffers import ReplayBuffer
from torchy_baselines.common.evaluation import evaluate_policy
from torchy_baselines.sac.policies import SACPolicy


class SAC(BaseRLModel):
    """
    Implementation of Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    Paper: https://arxiv.org/abs/1801.01290
    Code: This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    """

    def __init__(self, policy, env, policy_kwargs=None, verbose=0,
                 buffer_size=int(1e6), learning_rate=3e-4, seed=0, device='auto',
                 ent_coef='auto', target_entropy='auto', gamma=0.99,
                 action_noise_std=0.0, start_timesteps=100,
                 batch_size=64, create_eval_env=False,
                 _init_setup_model=True):

        super(SAC, self).__init__(policy, env, SACPolicy, policy_kwargs, verbose, device,
                                  create_eval_env=create_eval_env)

        self.max_action = np.abs(self.action_space.high)
        self.action_noise_std = action_noise_std
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.start_timesteps = start_timesteps
        self._seed = seed
        self.batch_size = batch_size

        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.log_ent_coef = None
        # self.target_update_interval = target_update_interval
        # self.gradient_steps = gradient_steps
        self.gamma = gamma

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        obs_dim, action_dim = self.observation_space.shape[0], self.action_space.shape[0]
        self.seed(self._seed)

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == 'auto':
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if '_' in self.ent_coef:
                init_value = float(self.ent_coef.split('_')[1])
                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            self.ent_coef = th.exp(self.log_ent_coef.detach())
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef = float(self.ent_coef)

        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, action_dim, self.device)
        self.policy = self.policy(self.observation_space, self.action_space,
                                  self.learning_rate, device=self.device, **self.policy_kwargs)
        self.policy = self.policy.to(self.device)
        self._create_aliases()

    def _create_aliases(self):
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def select_action(self, observation):
        # Normally not needed
        observation = np.array(observation)
        with th.no_grad():
            observation = th.FloatTensor(observation.reshape(1, -1)).to(self.device)
            return self.actor(observation).cpu().data.numpy()

    def predict(self, observation, state=None, mask=None, deterministic=True):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        return self.max_action * self.select_action(observation)

    def train(self, n_iterations, batch_size=64, tau=0.005):

        for it in range(n_iterations):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            obs, action_batch, next_obs, done, reward = replay_data

            # Action by the current actor for the sampled state
            action_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if not isinstance(self.ent_coef, float):
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Select action according to policy
            next_action, next_log_prob = self.actor.action_log_prob(next_obs)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = th.min(target_q1, target_q2)
            target_q = reward + ((1 - done) * self.gamma * target_q).detach()

            # td error + entropy term
            q_backup = (target_q - self.ent_coef * next_log_prob.reshape(-1, 1)).detach()

            # Get current Q estimates
            # using action from the replay buffer
            current_q1, current_q2 = self.critic(obs, action_batch)

            # Compute critic loss
            critic_loss = 0.5 * (F.mse_loss(current_q1, q_backup) + F.mse_loss(current_q2, q_backup))

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - min_qf_pi)
            actor_loss = (self.ent_coef * log_prob - self.critic.q1_forward(obs, action_pi)).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self, total_timesteps, callback=None, log_interval=100,
              eval_env=None, eval_freq=-1, n_eval_episodes=5, tb_log_name="TD3", reset_num_timesteps=True):

        timesteps_since_eval = 0
        episode_num = 0
        evaluations = []
        start_time = time.time()
        eval_env = self._get_eval_env(eval_env)

        while self.num_timesteps < total_timesteps:

            if callback is not None:
                # Only stop training if return value is False, not when it is None.
                if callback(locals(), globals()) is False:
                    break

            episode_reward, episode_timesteps = self.collect_rollouts(self.env, n_episodes=1,
                                                                      action_noise_std=self.action_noise_std,
                                                                      deterministic=False, callback=None,
                                                                      start_timesteps=self.start_timesteps,
                                                                      num_timesteps=self.num_timesteps,
                                                                      replay_buffer=self.replay_buffer)
            episode_num += 1
            self.num_timesteps += episode_timesteps
            timesteps_since_eval += episode_timesteps

            if self.num_timesteps > 0:
                if self.verbose > 1:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                        self.num_timesteps, episode_num, episode_timesteps, episode_reward))
                self.train(episode_timesteps, batch_size=self.batch_size)

            # Evaluate episode
            if 0 < eval_freq <= timesteps_since_eval and eval_env is not None:
                timesteps_since_eval %= eval_freq
                mean_reward, _ = evaluate_policy(self, eval_env, n_eval_episodes)
                evaluations.append(mean_reward)
                if self.verbose > 0:
                    print("Eval num_timesteps={}, mean_reward={:.2f}".format(self.num_timesteps, evaluations[-1]))
                    print("FPS: {:.2f}".format(self.num_timesteps / (time.time() - start_time)))

        return self

    def save(self, path):
        if not path.endswith('.pth'):
            path += '.pth'
        th.save(self.policy.state_dict(), path)

    def load(self, path, env=None, **_kwargs):
        if not path.endswith('.pth'):
            path += '.pth'
        if env is not None:
            pass
        self.policy.load_state_dict(th.load(path))
        self._create_aliases()
