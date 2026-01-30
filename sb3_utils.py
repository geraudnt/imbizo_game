import os, copy, numpy as np, torch, torch.nn as nn, gymnasium as gym
from typing import Any, Dict, List, Optional, Union
from sm import BaseAgent, evaluate

from stable_baselines3 import HerReplayBuffer, DQN, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


class WVFReplayBuffer(HerReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_info_dict = True

    def _get_virtual_samples(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Get the samples, sample new desired goals and compute new rewards.

        :param batch_indices: Indices of the transitions
        :param env_indices: Indices of the environments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples, with new desired goals and new rewards
        """
        # Get infos and obs
        obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
        # Sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # The desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals
        
        assert (
            self.env is not None
        ), "You must initialize HerReplayBuffer with a VecEnv so it can compute rewards for virtual transitions"
        
        infos = self.infos[batch_indices, env_indices]
        done_actions = (next_obs["achieved_goal"]==obs["desired_goal"]).min(1)
        env_rewards, dones = [], [] 
        for i in range(len(infos)): 
            env_rewards.append(infos[i]["env_reward"])
            dones.append(infos[i]["env_done"] or done_actions[i])
        env_rewards, dones = np.array(env_rewards), np.array(dones)
        # dones = self.dones[batch_indices, env_indices]

        # Compute new reward
        rewards = self.env.env_method("compute_reward", next_obs["achieved_goal"], obs["desired_goal"], env_rewards, dones, indices=[0],)
        rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        # obs = self._normalize_obs(obs, env)  # type: ignore[assignment]
        # next_obs = self._normalize_obs(next_obs, env)  # type: ignore[assignment]
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}
        
        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations=next_observations,
            dones=self.to_torch(dones).reshape(-1, 1),
            rewards=self.to_torch(rewards).reshape(-1, 1),
            # dones=self.to_torch(dones * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            # rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),  # type: ignore[attr-defined]
        )

class UVFAFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.features = ["violated_constraints", "desired_goal"] if "violated_constraints" in observation_space else sorted([k for k in observation_space.keys() if k!="env_state"])

        c = observation_space["env_state"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad(): cnn_embedding = self.cnn(torch.as_tensor(observation_space["env_state"].sample()[None]).float()).shape[1]
        mlp_input = np.sum([observation_space[k].shape[0] for k in self.features])
        self.mlp = nn.Sequential(nn.Linear(mlp_input, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(cnn_embedding + features_dim, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU(), nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations):
        cnn_embedding = self.cnn(observations["env_state"])
        mlp_embedding = self.mlp(torch.cat([observations[k] for k in self.features], dim=1))
        return self.linear(torch.cat([cnn_embedding, mlp_embedding], dim=1))


class EvaluateSaveCallback(BaseCallback):
    def __init__(self, primitive_env, task_env, SM=None, skill=None, save_dir="", eval_steps=1000, print_freq=1e4, seed=None, verbose=1):
        super().__init__(verbose)
        self.primitive_env, self.task_env, self.SM, self.skill, self.eval_steps, self.print_freq, self.save_dir, self.seed  = primitive_env, task_env, SM, skill, eval_steps, print_freq, save_dir, seed
        self.rewards, self.successes, self.best = 0, 0, 0
        
    def _on_step(self) -> bool:        
        if (self.n_calls-1) % self.print_freq == 0:
            if self.task_env: self.rewards, self.successes, _ = evaluate(self.task_env, SM=self.SM, skill=self.skill, gamma=0.99, eval_steps=self.eval_steps, seed=self.seed) 
            else:             self.rewards, self.successes, _ = np.sum(self.primitive_env.rewards), np.sum(self.primitive_env.successes)/100
            if self.rewards >= self.best:
                self.best = self.rewards
                if self.SM:    self.model.save(self.save_dir+"wvf_"+self.primitive_env.primitive)   
                if self.skill: self.model.save(self.save_dir+"skill")   
                if self.primitive_env: torch.save(self.primitive_env.goals, self.save_dir+"goals") 
        if self.task_env:      self.logger.record("eval total reward", self.rewards); self.logger.record("eval successes", self.successes)
        else:                  self.logger.record("total reward", self.rewards); self.logger.record("successes", self.successes)
        if self.primitive_env: self.logger.record("goals", len(self.primitive_env.goals))
        return True


class DQNAgent(BaseAgent):
    def __init__(self, name, env, save_dir=None, log_dir=None, load=False, buffer_size=1000000, use_her=16):
        self.name, self.action_space, self.observation_space = name, env.action_space, env.observation_space
        self.model_class = DQN
        self.model = self.model_class(
                    "MultiInputPolicy", env, verbose = 1,
                    policy_kwargs = dict(features_extractor_class = UVFAFeaturesExtractor, features_extractor_kwargs = dict(features_dim=1024)),
                    replay_buffer_class = None if not use_her else WVFReplayBuffer,
                    replay_buffer_kwargs = None if not use_her else dict(n_sampled_goal = use_her, goal_selection_strategy = "future", ),
                    # learning_rate = 1e-5, gamma = 0.99, learning_starts = 10000, target_update_interval = 1000, train_freq = 1,
                    exploration_fraction = 0.5, exploration_final_eps = 0.1, buffer_size=buffer_size
                )
        if log_dir: self.model.set_logger(configure(log_dir+self.name, ["stdout", "csv", "tensorboard"]))
        if load: self.model = self.model_class.load(save_dir+self.name, env=env, buffer_size=buffer_size)
        os.makedirs(save_dir, exist_ok=True)

    def get_action_value(self, states):
        obs = states.copy()
        for key,obs_ in obs.items():
            obs[key] = torch.as_tensor(obs_, device=self.model.device)
            if len(obs[key].shape)>2: obs[key] =  obs[key].permute(0, 3, 1, 2)
        with torch.no_grad(): values = self.model.q_net(obs)
        if values.device != torch.device("cpu"): values = values.cpu()
        # print("deterministic action:", values.numpy().argmax(1), "stochastic action:", self.model.predict(states, deterministic=False)[0].squeeze())
        return values.numpy().argmax(1), values.numpy().max(1)
        

class TD3Agent(BaseAgent):
    def __init__(self, name, env, save_dir=None, log_dir=None, load=False, buffer_size=1000000, use_her=0):
        self.name, self.action_space, self.observation_space = name, env.action_space, env.observation_space
        self.model_class = TD3
        self.model = self.model_class(
                    "MultiInputPolicy", env, verbose=1, 
                    policy_kwargs=dict(net_arch=[2024, 2024, 2024]),
                    replay_buffer_class = None if not use_her else WVFReplayBuffer,
                    replay_buffer_kwargs = None if not use_her else dict(n_sampled_goal = use_her, goal_selection_strategy = "future", ),
                    action_noise = NormalActionNoise(mean=np.zeros(self.action_space.shape[-1]), sigma=0.2 * np.ones(self.action_space.shape[-1])),
                    learning_rate=1e-5, gamma=0.99, batch_size=32, learning_starts=1000, train_freq=50, gradient_steps=50, buffer_size=buffer_size
                )
        if log_dir: self.model.set_logger(configure(log_dir+self.name, ["stdout", "csv", "tensorboard"]))
        if load: self.model = self.model_class.load(save_dir+self.name, env=env, buffer_size=buffer_size)
        os.makedirs(save_dir, exist_ok=True)
    
    def get_action_value(self, states):
        obs = states.copy()
        for key,obs_ in obs.items():
            obs[key] = torch.as_tensor(obs_, device=self.model.device)
            if len(obs[key].shape)>2: obs[key] = obs[key].permute(0, 3, 1, 2)
        with torch.no_grad(): 
            actions = self.model.actor(obs).clamp(-1, 1)
            values = self.model.critic(obs, actions)[0]
        if values.device != torch.device("cpu"): actions, values = actions.cpu(), values.cpu()
        # print("deterministic action:", actions.squeeze().numpy(), "stochastic action:", self.model.predict(states, deterministic=False)[0].squeeze())
        return actions.squeeze().numpy(), values.squeeze().numpy()
