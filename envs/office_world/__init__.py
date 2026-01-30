"""
Example task distributions
"""

import sys
sys.path.insert(1, '../../')

import numpy as np
import gymnasium as gym
from rm import Task


class OfficeCoffeeTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Deliver coffee to the office without breaking decorations"""
        env = gym.make("Office-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (c & X (F o))) & (G~d)", **kwargs)
        super().__init__(task)

class OfficePatrolTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Patrol rooms A, B, C, and D without breaking any decoration"""
        env = gym.make("Office-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (A & X (F (B & X (F (C & X (F D))))))) & (G~d)", **kwargs)
        super().__init__(task)

class OfficeCoffeeMailTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Deliver coffee and mail to the office without breaking any decoration"""
        env = gym.make("Office-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "((F (m & X (F (c & X (F o))))) | (F (c & X (F (m & X (F o)))))) & (G~d)", **kwargs)
        super().__init__(task)

class OfficeLongTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Deliver mail to the office until there is no mail left, then deliver coffee to office while there are people in the office, then patrol rooms A-B-C-D-A, and never break a decoration"""
        env = gym.make("Office-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (m & X (F (o & X (~ m U (~ tm & m & X (F (c & X (~ o U (~ to & o & X (F (A & X (F (B & X (F (C & X (F (D & X (F A))))))))))))))))))) & (G~d)", **kwargs)
        super().__init__(task)

class OfficeMultiTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Randomly sampled task distribution"""
        self.tasks = ["(F (c & X (F o))) & (G~d)","(F (A & X (F (B & X (F (C & X (F D))))))) & (G~d)","(F (m & X (F (o & X (~ m U (~ tm & m & X (F (c & X (~ o U (~ to & o & X (F (A & X (F (B & X (F (C & X (F (D & X (F A))))))))))))))))))) & (G~d)"]
        env = gym.make("Office-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        self.create_task = lambda : Task(env, np.random.choice(self.tasks), **kwargs)
        super().__init__(self.create_task())

    def reset(self, *args, **kwargs):
        super().__init__(self.create_task())
        state, info = self.unwrapped.reset(**kwargs)
        return state, info

class OfficeModCoffeeTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Deliver coffee to the office without breaking decorations"""
        env = gym.make("Office-Mod-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "(F (c & X (F o))) & (G~d)", **kwargs)
        super().__init__(task)

class OfficeModCoffeeStrictTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Deliver coffee to the office and do not go to office until you have coffee"""
        env = gym.make("Office-Mod-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "(F o) & (~o U c)", **kwargs)
        super().__init__(task)


gym.envs.registration.register(
    id='Office-v0',
    entry_point='envs.office_world.GridWorld:GridWorldEnv',
)
gym.envs.registration.register(
    id='Office-Mod-v0',
    entry_point='envs.office_world.GridWorld_modified:GridWorldEnv',
)

gym.envs.registration.register(
    id='Office-Coffee-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeCoffeeTask,
)
gym.envs.registration.register(
    id='Office-Patrol-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficePatrolTask,
)
gym.envs.registration.register(
    id='Office-CoffeeMail-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeCoffeeMailTask,
)
gym.envs.registration.register(
    id='Office-Long-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeLongTask,
)
gym.envs.registration.register(
    id='Office-Multi-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeMultiTask,
)
gym.envs.registration.register(
    id='Office-Mod-Coffee-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeModCoffeeTask,
)
gym.envs.registration.register(
    id='Office-Mod-CoffeeStrict-Task-v0',
    max_episode_steps=1000, 
    entry_point=OfficeModCoffeeStrictTask,
)
