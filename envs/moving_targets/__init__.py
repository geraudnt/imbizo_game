import sys
sys.path.insert(1, '../../')

import numpy as np, random
import gymnasium as gym
from rm import Task


class BlueTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up a blue object. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "G (F blue)", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)
        
class PurpleTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up a purple object. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "G (F purple)", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)
        
class SquareTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up any object. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, "G (F square)", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)

class Task1(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up any object. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, hoa="envs/moving_targets/MovingTargets-Task-1-v0.hoa", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)

class Task2(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up blue then purple objects, then objects that are neither blue nor purple. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, hoa="envs/moving_targets/MovingTargets-Task-2-v0.hoa", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)

class Task3(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up blue objects or squares, but never blue squares. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, hoa="envs/moving_targets/MovingTargets-Task-3-v0.hoa", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)

class Task4(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, **kwargs):
        """Pick up non-square blue objects, then nonblue squares in that order. Repeat this forever."""
        env = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        task = Task(env, hoa="envs/moving_targets/MovingTargets-Task-4-v0.hoa", accept_terminal=False, rmax=2, **kwargs)
        super().__init__(task)

class MultiTask(gym.Wrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, *args, **kwargs):
        """Randomly sampled task distribution"""
        self.kwargs = kwargs
        self.environment = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        self.tasks = ["F (circle | square)",
                      "F(blue & X(F(purple & X(F(((circle | square) & ~(blue | purple)))))))",
                      "(F(blue | square)) & (G ~(blue & square))",
                      "F((~square & blue) & X(F(square & ~ blue)))"
                      ]
        super().__init__(self.create_task())
    
    def create_task(self):
        self.idx = np.random.choice(len(self.tasks))
        return Task(self.environment, self.tasks[self.idx], max_states=10, **self.kwargs)
    
    def reset(self, *args, **kwargs):
        super().__init__(self.create_task())
        return self.env.reset(**kwargs)

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
        self.environment = gym.make("MovingTargets-v0", render_mode=kwargs['render_mode'] if "render_mode" in kwargs else "human")
        
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
    id='CollectEnv-v0',
    entry_point='envs.moving_targets.moving_targets:CollectEnv',
)
gym.envs.registration.register(
    id='MovingTargets-v0',
    entry_point='envs.moving_targets.moving_targets:MovingTargets',
)

gym.envs.registration.register(
    id='MovingTargets-Blue-Task-v0',
    max_episode_steps=100, 
    entry_point=BlueTask,
)
gym.envs.registration.register(
    id='MovingTargets-Purple-Task-v0',
    max_episode_steps=100, 
    entry_point=PurpleTask,
)
gym.envs.registration.register(
    id='MovingTargets-Square-Task-v0',
    max_episode_steps=100, 
    entry_point=SquareTask,
)
gym.envs.registration.register(
    id='MovingTargets-Task-1-v0',
    max_episode_steps=100, 
    entry_point=Task1,
)
gym.envs.registration.register(
    id='MovingTargets-Task-2-v0',
    max_episode_steps=100, 
    entry_point=Task2,
)
gym.envs.registration.register(
    id='MovingTargets-Task-3-v0',
    max_episode_steps=100, 
    entry_point=Task3,
)
gym.envs.registration.register(
    id='MovingTargets-Task-4-v0',
    max_episode_steps=100, 
    entry_point=Task4,
)
gym.envs.registration.register(
    id='MovingTargets-Multi-Task-v0',
    max_episode_steps=100, 
    entry_point=MultiTask,
)
gym.envs.registration.register(
    id='MovingTargets-PartiallyOrdered-Task-v0',
    max_episode_steps=100, 
    entry_point=PartiallyOrderedTask,
)
