import os, argparse, imageio, random, time, torch, numpy as np, gymnasium as gym, envs
from PIL import Image
from sympy import sympify, Symbol

from sm import TaskPrimitive, SkillMachine
from sb3_utils import TD3Agent, DQNAgent
from sm_ql import QLAgent
from rm import Task


def render(state, SM, skill, task_env, images, max_width, max_height):
    states = {k:np.expand_dims(v,0) for (k,v) in state.items()}
    if hasattr(task_env.environment,"render_skill"): task_env.environment.render_params["skill"] = skill
    if hasattr(task_env.environment,"render_params"):
        task_env.environment.render_params["state"] = states
        task_env.environment.render_params["task_title"] = r"Task: ${}$".format(str(task_env.task).replace("&"," \wedge ").replace("|"," \\vee ").replace("~"," \\neg "))
        exp = sympify(SM.exp); exp = exp if type(exp)==bool else exp.subs({symbol: Symbol(' Q_{{{}}} '.format(symbol)) for symbol in exp.free_symbols})
        task_env.environment.render_params["skill_title"] = r"SM($u_{{{}}}$): ${}$".format(task_env.rm.u, str(exp).replace("&"," \wedge ").replace("|"," \\vee ").replace("~"," \\neg ").replace("p_","").replace("c_","\widehat ").replace("True","Q_{True}").replace("False","Q_{False}"))
    images.append(task_env.render())
    if task_env.render_mode =="rgb_array":
        if images[-1].shape[0]>max_width: max_width = images[-1].shape[0]
        if images[-1].shape[1]>max_height: max_height = images[-1].shape[1]
    
    return images, max_width, max_height


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="The RM augmented environment for a predefined task distribution.", default='Office-Coffee-Task-v0')
parser.add_argument("--algo", help="RL algorithm", default='ql')
parser.add_argument("--ltl", help="The ltl task the agent should solve. Ignored if --env is a task.", default='')
parser.add_argument("--fewshot", help="Load pretrained skill primitives", action='store_true', default=False)
parser.add_argument("--sp_dir", help="Directory where the learned skill primitives will be saved", default='')
parser.add_argument("--q_dir", help="Directory where the learned task specific skill will be saved", default='')
parser.add_argument("--save_path", help="Save renders to path.", default='')
parser.add_argument("--render_mode", help="Render mode. E.g. human, rgb_array", default="rgb_array")
parser.add_argument("--episodes", help="Number of episodes", type=int, default=1)
parser.add_argument("--seed", help="Random seed", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    gym.logger.set_level(gym.logger.ERROR) 
    q_dir = args.q_dir if args.q_dir else f'./data/{args.algo}_vf/{args.env}/'
    sp_dir = args.sp_dir if args.sp_dir else f'./data/{args.algo}_wvf/{args.env}/'
  
    # Initialise the MDP for the given temporal logic task
    if "Task" in args.env: task_env = gym.make(args.env, render_mode=args.render_mode, test=True)
    else:                  task_env = Task(gym.make(args.env, max_episode_steps=500, render_mode=args.render_mode), args.ltl, test=True)
    if hasattr(task_env.environment.unwrapped, "start_position"): task_env.environment.unwrapped.start_position = (10,4)

    # Load the zeroshot or fewshot skill
    if args.fewshot:
        skill = torch.load(q_dir+"skill"); SM = skill.SM
        save_path = args.save_path if args.save_path else f"./images/fewshot_{args.env}_{args.ltl}.gif"
    else:
        primitive_env = TaskPrimitive(task_env.environment, sb3=args.algo!="ql")
        primitive_env.goals.update(torch.load(sp_dir+"goals")) 
        if args.algo=="ql":    SP = {primitive: QLAgent(primitive, primitive_env, save_dir=sp_dir, load=True) for primitive in ['0','1']}
        elif args.algo=="dqn": SP = {primitive: DQNAgent("wvf_"+primitive, primitive_env, save_dir=sp_dir, load=True, buffer_size=1) for primitive in ['0','1']}
        elif args.algo=="td3": SP = {primitive: TD3Agent("wvf_"+primitive, primitive_env, save_dir=sp_dir, load=True, buffer_size=1) for primitive in ['0','1']}
        SM = SkillMachine(primitive_env, SP, vectorised=args.algo!="ql"); skill = SM
        save_path = args.save_path if args.save_path else f"./images/zeroshot_{args.env}_{args.ltl}.gif"
    print("save_path", save_path)

    # Visualise the skill solving the given task
    successes, episode, all_images, max_width, max_height = 0, 0, [], 0, 0
    while episode<args.episodes:
        episode += 1; step = 0; rewards = 0; images = []
        state, info = task_env.reset(seed=args.seed)
        SM.reset(task_env.rm, info["true_propositions"])
        images, max_width, max_height = render(state, SM, skill, task_env, images, max_width, max_height)
        
        while True:
            states = {k:np.expand_dims(v,0) for (k,v) in state.items()}
            action = skill.get_action_value(states)[0][0] if np.random.random() > 0.0 else task_env.action_space.sample()
            state_, reward, done, truncated, info = task_env.step(action)       
            SM.step(task_env.rm, info["true_propositions"])

            step += 1; rewards += reward
            images, max_width, max_height = render(state, SM, skill, task_env, images, max_width, max_height)
            print("episode", episode, "step", step, "skill", SM.exp, "rewards", rewards, "props", info["true_propositions"], "rm_state", task_env.rm.u)
            
            # if done or (state["env_state"]==state_["env_state"]).all() or truncated: successes += reward>=task_env.rm.rmax; break
            if done or truncated or reward>0: successes += reward>=task_env.rm.rmax; break
            state = state_
            
        if rewards>=task_env.rm.rmax: all_images += images
        else:                        episode -= 1
    if args.render_mode=="rgb_array": 
        all_images = [np.pad(image, pad_width = [(max(0,d2-d1)//2, max(0,d2-d1) - max(0,d2-d1)//2) 
                                                    for d1, d2 in zip(image.shape, (max_width,max_height,3))
                                                ], mode='constant', constant_values=0) 
                        for image in all_images
                    ]
        imageio.mimsave(save_path, all_images, loop=0, fps=10)
