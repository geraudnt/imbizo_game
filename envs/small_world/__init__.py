"""
Example task distributions
"""

import sys
sys.path.insert(1, '../../')

import numpy as np
import gymnasium as gym
from rm import Task


gym.envs.registration.register(
    id='Small-v0',
    entry_point='envs.small_world.GridWorld:GridWorldEnv',
)