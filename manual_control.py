#!/usr/bin/env python3

"""
Code derived from Maxime et al: https://github.com/maximecb/gym-minigrid
"""
# For Mujoco envs, to render rgb: export LD_PRELOAD=""
# For Mujoco envs, to render render regular window: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

import argparse, numpy as np, cv2
import gymnasium as gym, envs
from sm import TaskPrimitive
from rm import Task

vars = dict(title="", done_action=False)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def redraw(img):
    if not args.agent_view: img = env.render()
    img = cv2.resize(img, dsize=(1024,int(1024*(img.shape[0]/img.shape[1]))), interpolation=cv2.INTER_AREA)
    img = transform_rgb_bgr(img); shape = img.shape[1]
    violet = np.ones((img.shape[0], args.window_width, 3), np.uint8)*255
    img = cv2.hconcat((img,violet))
    for i,title in enumerate(vars["title"].split("\n")): cv2.putText(img, title, (shape,100+75*i), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4, 0)
    cv2.imshow(args.env, img)   

def start():
    reset()
    while True:
        key = cv2.waitKey(0)
        key_handler(key)
        if key == -1 or key == 27: cv2.destroyAllWindows(); break

def reset():      
    vars["done_action"] = False
    obs, _ = env.reset(seed=args.seed)
    if type(obs)==dict: obs = obs["env_state"]
    if "Task" in args.env or args.ltl: print(env.rm.rm_str)

    predicates = env.get_predicates()
    env_state = obs if len(obs.shape)==1 and obs.shape[0] < 10 else f"shape {obs.shape}"
    vars["title"] = ' Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(env.predicates, env.constraints,env_state,predicates,None,None,None)
    if hasattr(env, "task"): vars["title"] = ' Task = {} \n Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(env.task, env.predicates, env.constraints,env_state,predicates,None,None,None)
    if args.primitive:       vars["title"] = ' Primitive = {} \n Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(args.primitive, env.predicates, env.constraints,env_state,predicates,None,None,None)
    
    redraw(obs)

def step(action):
    obs, reward, done, truncated, _ = env.step(action)
    if type(obs)==dict: obs = obs["env_state"]
    
    predicates = env.get_predicates()
    done_ = False if not done else "True (Press 'backspace' to restart)"
    truncated_ = False if not truncated else "True (Press 'backspace' to restart)"
    env_state = obs if len(obs.shape)==1 and obs.shape[0] < 10 else f"shape {obs.shape}"
    vars["title"] = ' Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(env.predicates, env.constraints,env_state,predicates,reward,done_,truncated_)
    if hasattr(env, "task"): vars["title"] = ' Task = {} \n Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(env.task, env.predicates, env.constraints,env_state,predicates,reward,done_,truncated_)
    if args.primitive:       vars["title"] = ' Primitive = {} \n Propositions =  {} \n Constraints = {} \n Observation = {} \n True propositions = {} \n Reward = {} \n Done = {} \n Truncated = {} '.format(args.primitive, env.predicates, env.constraints,env_state,predicates,reward,done_,truncated_)

    redraw(obs)

def key_handler(key):
    keys = {27:'escape', 8:'backspace', 13:'enter', 32:' ', 81:'left', 82:'up', 83:'right', 84:'down'}
    if key not in keys: print('Unknown pressed', key); return
    key = keys[key]
    print('pressed', key)
    if key == 'backspace': reset()  
    elif key == 'enter': vars["done_action"] = True  
    elif key == ' ': step(env.action_space.sample())
    elif hasattr(env,"actions") and type(env.actions)==dict and key in env.actions: 
        action = env.actions[key]
        if vars["done_action"] and hasattr(env,"primitive"):
            if not env.stable_baselines: 
                if type(env.action_space) == gym.spaces.Box: action[-1] = env.action_space.max()
                else:                                        action = action % env.environment.action_space.n + env.environment.action_space.n
        step(action)
    
parser = argparse.ArgumentParser()
parser.add_argument("--env", help="the agent's environment. It can also be the RM augmented environment for a predefined task distribution.", default='Office-v0')
parser.add_argument("--primitive", help="the task primitive the agent should solve. E.g. p_A. Leave empty to show the regular environment.", default="")
parser.add_argument("--ltl", help="the ltl task the agent should solve. E.g. (F (c & X (F o))) & (G~d). Leave empty to show the regular environment.", default="")
parser.add_argument("--window_width", help="Render mode. E.g. human, rgb_array", type=int, default=2000)
parser.add_argument("--seed", type=int, help="random seed to generate the environment with", default=None)
parser.add_argument("--agent_view", help="Render agent observation", action='store_true', default=False)
args = parser.parse_args()

if __name__ == "__main__":
    gym.logger.set_level(gym.logger.ERROR) 
    
    env = gym.make(args.env, render_mode = "rgb_array")
    if not hasattr(env, "constraints"): env.constraints=env.predicates
    if args.primitive: env = TaskPrimitive(env, args.primitive, wvf=False, render_mode = "rgb_array")
    elif args.ltl:     env = Task(env, args.ltl, render_mode = "rgb_array")
    print("Observation space: ", env.observation_space)
    
    cv2.namedWindow(args.env, cv2.WINDOW_NORMAL); start()