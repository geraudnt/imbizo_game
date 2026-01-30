#!/usr/bin/env python3

"""
Code mostly derived from Maxime et al: https://github.com/maximecb/gym-minigrid
"""

import time
import argparse
import numpy as np
import gym
from GridWorld import *
from window import Window

def redraw():
    img = env.render(env_map=True, mode='rgb_array')

    # window.fig.savefig("images/office_world.pdf")
    window.show_img(img)    

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
    print('obs={}, reward={}'.format(env.goals[obs] if type(obs)==int else obs, reward))
    print("Events: ",env.get_events())

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
    if event.key == 'enter':
        step(env.actions['DONE'])
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    help="gym environment to load",
    default='coffee'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)

args = parser.parse_args()

gridworld_objects =  {
    '1room': roomA(),
    '2room': roomB(),
    '3room': roomC(),
    '4room': roomD(),
    'coffee': coffee(),
    'mail': mail(3),
    'office': office(9),
    'decor': decor(),
}
env = GridWorld(gridworld_objects=gridworld_objects)

env = Task(env)
#task_goals = []
#if args.task == "coffee":
#    for goal in range(env.goal_space.n):
#        if ('coffee' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "mail":
#    for goal in range(env.goal_space.n):
#        if ('mail' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "office":
#    for goal in range(env.goal_space.n):
#        if ('office' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "decor":
#    for goal in range(env.goal_space.n):
#        if ('decor' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "1room":
#    for goal in range(env.goal_space.n):
#        if ('1room' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "2room":
#    for goal in range(env.goal_space.n):
#        if ('2room' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "3room":
#    for goal in range(env.goal_space.n):
#        if ('3room' in env.goals[goal]):
#            task_goals.append(goal)
#if args.task == "4room":
#    for goal in range(env.goal_space.n):
#        if ('4room' in env.goals[goal]):
#            task_goals.append(goal)

#env = Task(env, task_goals=task_goals)
print(len(env.possiblePositions), len(env.goals), len(env.possiblePositions)*len(env.goals))
print(env.predicate_keys)

window = Window('Office-World: ' + args.task)
window.reg_key_handler(key_handler)
window.fig.set_size_inches(6, 3)

reset()

# Blocking event loop
window.show(block=True)
