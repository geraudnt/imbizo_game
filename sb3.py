import argparse, numpy as np, torch, gymnasium as gym, envs
from sb3_utils import EvaluateSaveCallback, TD3Agent, DQNAgent
from rm import Task


parser = argparse.ArgumentParser()
parser.add_argument("--env", help="The agent's environment, or the RM augmented environment for a predefined task distribution.", default='Safety-v0')
parser.add_argument("--ltl", help="The ltl task the agent should solve. Ignored if --env is a task.", default='')
parser.add_argument("--total_steps", help="Total training steps", type=int, default=1000000)
parser.add_argument("--eval_steps", help="Max steps per eval episode", type=int, default=1000)
parser.add_argument("--algo", help="Stable baselines3 algorithm", default='td3')
parser.add_argument("--skill_dir", help="Directory where the learned skill will be saved", default='')
parser.add_argument("--log_dir", help="Directory where the results will be saved", default='')
parser.add_argument("--load", help="Load pretrained skill primitives", action='store_true', default=False)
if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = args.log_dir if args.log_dir else f'./data/logs/{args.algo}/{args.env}/'
    skill_dir = args.skill_dir if args.skill_dir else f'./data/{args.algo}/{args.env}/'
    assert ("Task" in args.env) or args.ltl, "A task distribution or ltl specification must be provided"

    # Initialise MDP for the given temporal logic task (task_env)
    if "Task" in args.env: 
        task_env = gym.make(args.env)
        task_env_test = gym.make(args.env, test=True)
    else:
        task_env = Task(gym.make(args.env), args.ltl)
        task_env_test = Task(gym.make(args.env), args.ltl, test=True)
   
    # Initialise task specific skill
    if args.algo=="dqn":   agent = DQNAgent("skill", task_env, skill_dir, log_dir, args.load, use_her=0)
    elif args.algo=="td3": agent = TD3Agent("skill", task_env, skill_dir, log_dir, args.load, use_her=0)

    # Start Training
    agent.model.learn(args.total_steps, EvaluateSaveCallback(None, task_env_test, None, agent, skill_dir, args.eval_steps))
