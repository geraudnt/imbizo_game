"""
Example task distributions
"""

import sys
sys.path.insert(1, '../../')

import numpy as np, random
import gymnasium as gym
from rm import Task


d = 1.4

class Task1(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a button and then to a cylinder."""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (buttons & X (F goal)))", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 0
        # state, _, _, _, info = self.env.step(self.action_space.sample()*0)
        return state, info

class Task2(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a button and then to a cylinder while never entering blue regions"""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (buttons & X (F goal))) & (G~hazards)", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 0
        return state, info

class Task3(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a button, then to a cylinder without entering blue regions, then to a button inside a blue region, and finally to a cylinder again."""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "F(buttons & X(F (( goal & ~hazards )&X(F(( buttons & hazards )&X(F goal))))))", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 0
        return state, info
        
    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        if self.rm.u == 2: self.environment.unwrapped.env.goal_button = 1
        return state, reward, done, truncated, info
    
class Task4(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a button and then to a cylinder in a blue region."""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(0, -d-1)], 'robot_rot': 180*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (buttons & X (F (goal & hazards))))", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 2
        return state, info
    
class Task5(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a cylinder, then to a button in a blue region, and finally to a cylinder again."""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(0.5, -d-1)], 'robot_rot': 180*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (goal & X (F ((buttons & hazards) & X (F goal)))))", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 1
        return state, info
        
    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        if self.rm.u == 3: self.environment.unwrapped.env.goal_button = 0
        return state, reward, done, truncated, info

class Task6(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, test=False, **kwargs):
        """Navigate to a blue region, then to a button with a cylinder, and finally to a cylinder while avoiding blue regions."""
        config = {} if not test else {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d, -d)], 'robot_rot': 90*np.pi/180}
        env = gym.make("Safety-v0", config=config, render_mode=kwargs["render_mode"] if "render_mode" in kwargs else "human")
        task = Task(env, "F(hazards & X(F(( buttons & goal) & X ((F goal) & (G ~hazards)))))", **kwargs)
        super().__init__(task)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.environment.unwrapped.env.goal_button = 1
        return state, info
        
    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        if self.rm.u == 2: self.environment.unwrapped.env.goal_button = 0
        return state, reward, done, truncated, info

class MultiTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, test=False, **kwargs):
        """Randomly sampled task distribution"""
        self.tasks = ["(F (buttons & X (F goal)))",
                      "(F (buttons & X (F goal))) & (G~hazards)",
                      "F(buttons & X(F (( goal & ~hazards )&X(F(( buttons & hazards )&X(F goal))))))",
                      "(F (buttons & X (F (goal & hazards))))",
                      "(F (goal & X (F ((buttons & hazards) & X (F goal)))))",
                      "F(hazards & X(F(( buttons & goal) & X ((F goal) & (G ~hazards)))))"
                      ]
        self.configs = [{'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180},
                        {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180},
                        {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d+1, 0)], 'robot_rot': 180*np.pi/180},
                        {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(0, -d-1)], 'robot_rot': 180*np.pi/180},
                        {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(0.5, -d-1)], 'robot_rot': 180*np.pi/180},
                        {'continue_goal': False, 'hazards_locations': [(0,0), (0, d)], 'buttons_locations': [(-d, 0), (d, 0), (0, d), (0, -d)], 'robot_locations': [(d, -d)], 'robot_rot': 90*np.pi/180}
                        ]
        if test: self.configs = [{}]*len(self.tasks)
        self.kwargs = kwargs
        super().__init__(self.create_task())
    
    def create_task(self):
        self.idx = np.random.choice(len(self.tasks))
        env = gym.make("Safety-v0", config=self.configs[self.idx], render_mode=self.kwargs["render_mode"] if "render_mode" in self.kwargs else "human")
        return Task(env, self.tasks[self.idx], max_states=10, **self.kwargs)
    
    def reset(self, *args, **kwargs):
        super().__init__(self.create_task())
        state, info = self.env.reset(**kwargs)
        if self.idx == 0 or self.idx == 1 or self.idx == 2: self.environment.unwrapped.env.goal_button = 0
        elif self.idx == 3: self.environment.unwrapped.env.goal_button = 2
        elif self.idx == 4: self.environment.unwrapped.env.goal_button = 1
        elif self.idx == 5: self.environment.unwrapped.env.goal_button = 1
        return state, info
        
    def step(self, action):
        state, reward, done, truncated, info = super().step(action)
        if   self.idx == 2 and self.rm.u == 2: self.environment.unwrapped.env.goal_button = 1
        elif self.idx == 4 and self.rm.u == 3: self.environment.unwrapped.env.goal_button = 0
        elif self.idx == 5 and self.rm.u == 2: self.environment.unwrapped.env.goal_button = 0
        return state, reward, done, truncated, info

class PartiallyOrderedTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, num_conjuncts_range=(1,4), depth_range=(1,5), prob_disjunction=0.25, **kwargs):
        """
        The Partially ordered task distribution from the paper:
        Vaezipoor, Pashootan, et al. "Ltl2action: Generalizing ltl instructions for multi-task rl." International Conference on Machine Learning. PMLR, 2021.
        """
        self.kwargs = kwargs
        self.num_conjuncts_range, self.depth_range, self.prob_disjunction = num_conjuncts_range, depth_range, prob_disjunction
        config = {'hazards_locations': [(0,0)], 'buttons_locations': [(0, 0)]}
        self.environment = gym.make("Safety-v0", config=config, render_mode=self.kwargs["render_mode"] if "render_mode" in self.kwargs else "human")
        
        super().__init__(self.create_task())

    def reset(self, *args, **kwargs):
        super().__init__(self.create_task())
        return self.unwrapped.reset(**kwargs)
    
    def create_task(self):
        return Task(self.environment, self.generate_ltl_task(self.num_conjuncts_range, self.depth_range, self.prob_disjunction), max_states=100, **self.kwargs)

    def generate_ltl_task(self, num_conjuncts_range, depth_range, prob_disjunction):
        num_conjuncts = random.randint(*num_conjuncts_range)
        return " & ".join([self.generate_sequence(random.randint(*depth_range), prob_disjunction) for _ in range(num_conjuncts)])

    def generate_sequence(self, depth, prob_disjunction):
        if depth == 0: return random.choice(self.environment.predicates)
        if random.random() > 0.5: return f"F({self.generate_term(prob_disjunction)})"
        else:                     return f"F({self.generate_term(prob_disjunction)} & {self.generate_sequence(depth - 1, prob_disjunction)})"

    def generate_term(self, prob_disjunction):
        if random.random() < prob_disjunction: return f"{random.choice(self.environment.predicates)} | {random.choice(self.environment.predicates)}"
        else:                                  return f"{random.choice(self.environment.predicates)}"


gym.envs.registration.register(
    id='Safety-v0',
    entry_point='envs.safety_gym.safety_gym:SafetyEnv',
)
gym.envs.registration.register(
    id='Safety-Task-1-v0',
    max_episode_steps=200, 
    entry_point=Task1,
)
gym.envs.registration.register(
    id='Safety-Task-2-v0',
    max_episode_steps=200, 
    entry_point=Task2,
)
gym.envs.registration.register(
    id='Safety-Task-3-v0',
    max_episode_steps=500, 
    entry_point=Task3,
)
gym.envs.registration.register(
    id='Safety-Task-4-v0',
    max_episode_steps=500, 
    entry_point=Task4,
)
gym.envs.registration.register(
    id='Safety-Task-5-v0',
    max_episode_steps=500, 
    entry_point=Task5,
)
gym.envs.registration.register(
    id='Safety-Task-6-v0',
    max_episode_steps=500, 
    entry_point=Task6,
)
gym.envs.registration.register(
    id='Safety-Multi-Task-v0',
    max_episode_steps=500, 
    entry_point=MultiTask,
)
gym.envs.registration.register(
    id='Safety-PartiallyOrdered-Task-v0',
    max_episode_steps=500, 
    entry_point=PartiallyOrderedTask,
)
