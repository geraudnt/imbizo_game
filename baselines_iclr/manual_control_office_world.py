#!/usr/bin/env python3

"""
Code mostly derived from Maxime et al: https://github.com/maximecb/gym-minigrid
"""

import time
import argparse
import numpy as np
import gym
from envs.office_world_modified.GridWorld import *
from envs.office_world_modified.window import Window
from cmd_util import make_env

def redraw():
    if "new" in args.task:
        img = env.render(size=1.5) #env.render(env_map=True, mode='rgb_array')

        # window.fig.savefig("../images/office_world.pdf")
        window.show_img(img)    
    else:
        img = env.render()
        print(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw()

def step(action):
    obs, reward, done, info = env.step(action)
    print('obs={}, events={}, reward={}, RM_state={}'.format(obs, env.get_events(), reward, None if not hasattr(env, "current_u_id") else env.current_u_id))
    if "new" in args.task:
        print(env.predicate_letters)

    if done:
        print('done!')
        reset()
    else:
        redraw()

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if "new" in args.task:
        if event.key == 'left':
            step(env.actions['LEFT'])
            return
        if event.key == 'right':
            step(env.actions['RIGHT'])
            return
        if event.key == 'up':
            step(env.actions['UP'])
            return
        if event.key == 'down':
            step(env.actions['DOWN'])
            return
    else:
        if event.key == 'left':
            step(3)
            return
        if event.key == 'right':
            step(1)
            return
        if event.key == 'up':
            step(0)
            return
        if event.key == 'down':
            step(2)
            return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    help="task to load",
    default='Office-new-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)

args = parser.parse_args()

env = gym.make(args.task)
print(env.events)

window = Window('Office-World: ' + args.task)
window.reg_key_handler(key_handler)
window.fig.set_size_inches(6, 3)

reset()

# Blocking event loop
window.show(block=True)
