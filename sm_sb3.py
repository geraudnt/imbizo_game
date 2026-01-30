import argparse, numpy as np, torch, gymnasium as gym, envs
from sb3_utils import EvaluateSaveCallback, TD3Agent, DQNAgent
from sm import TaskPrimitive, SkillMachine
from rm import Task


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="The agent's environment, or the RM augmented environment for a predefined task distribution.", default='Safety-v0')
parser.add_argument("--ltl", help="The ltl task the agent should solve. Ignored if --env is a task.", default='')
parser.add_argument("--total_steps", help="Total training steps", type=int, default=1000000)
parser.add_argument("--eval_steps", help="Max steps per eval episode", type=int, default=1000)
parser.add_argument("--algo", help="Stable baselines3 algorithm", default='td3')
parser.add_argument("--sp_dir", help="Directory where the learned skill primitives will be saved", default='')
parser.add_argument("--log_dir", help="Directory where the results will be saved", default='')
parser.add_argument("--load", help="Load pretrained skill primitives", action='store_true', default=False)
if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = args.log_dir if args.log_dir else f'./data/logs/sp_{args.algo}/{args.env}/'
    sp_dir = args.sp_dir if args.sp_dir else f'./data/sp_{args.algo}/{args.env}/'

    # Initialise MDP for the primitives (primitive_envs), and MDP for the given temporal logic task distribution (task_env)
    if "Task" in args.env: 
        primitive_env = TaskPrimitive(gym.make(args.env).environment, sb3=True)
        eval_task_env = gym.make(args.env, test=True)
    else:                  
        primitive_env = TaskPrimitive(gym.make(args.env), sb3=True)
        eval_task_env = Task(gym.make(args.env), args.ltl) if args.ltl else None
    if args.load: primitive_env.goals.update(torch.load(sp_dir+"goals")) 
   
    # Initialise the skill primitives from the min ("0") and max ("1") WVFs, and the skill machine from the skill primitives
    if args.algo=="dqn":   SP = {primitive: DQNAgent("wvf_"+primitive, primitive_env, sp_dir, log_dir, args.load) for primitive in ["0","1"]}
    elif args.algo=="td3": SP = {primitive: TD3Agent("wvf_"+primitive, primitive_env, sp_dir, log_dir, args.load) for primitive in ["0","1"]}
    SM = SkillMachine(primitive_env, SP, vectorised=True) 

    # Start Training
    for primitive, sp in SP.items():
        primitive_env.primitive = primitive; total_steps = args.total_steps*(0.1 if primitive=="0" else 0.9)
        sp.model.learn(total_steps, EvaluateSaveCallback(primitive_env, eval_task_env, SM, None, sp_dir, args.eval_steps))
