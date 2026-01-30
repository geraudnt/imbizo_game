import os, argparse, random, time, torch, numpy as np, gymnasium as gym, envs
from stable_baselines3.common.logger import configure
from sm import BaseAgent, evaluate
from rm import Task


class QLAgent(BaseAgent):
    """Q-learning Agent"""
    
    def __init__(self, env, lr=0.5, gamma=0.9, qinit=0):
        self.values, self.lr, self.gamma, self.qinit = {}, lr, gamma, qinit
        self.action_space, self.observation_space = env.action_space, env.observation_space
        
    def get_action_value(self, state):
        state_ = gym.spaces.flatten(self.observation_space, state).tobytes()
        if state_ not in self.values: self.values[state_] = np.zeros(self.action_space.n) + self.qinit
        action, value = self.values[state_].argmax(), self.values[state_].max() 
        return np.array([action]), np.array([value])
        
    def get_values(self, state):
        state = gym.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        return self.values[state]    
    
    def update_values(self, state, action, reward, state_, done):
        state = gym.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        if done: self.values[state][action] += self.lr * (reward - self.values[state][action])
        else:    self.values[state][action] += self.lr * (reward + self.gamma*self.get_values(state_).max() - self.values[state][action])


def learn(task_env, total_steps, q_dir="vf", log_dir="logs", gamma=0.9, lr=0.1, epsilon=0.5, qinit=0, eval_episodes=100, print_freq=10000, seed=None):  
    """Q-Learning based method for solving temporal logic tasks zeroshot or fewshot using Skill Machines"""

    # Initialise task specific value function
    Q = QLAgent(task_env, lr=lr, gamma=gamma, qinit=qinit)

    # Start Training
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    step, reward_total, successes, best_total_reward, num_episodes, start_time = 0, 0, 0, 0, 1, time.time()
    while step < total_steps:
        state, info = task_env.reset(seed=seed)       
        while True:            
            # Selecting and executing the action
            if random.random() < epsilon: action = task_env.action_space.sample()
            else:                         action = random.choice([action for action in range(task_env.action_space.n) if Q.get_values(state)[action] == Q.get_values(state).max()])
            state_, reward, done, truncated, info = task_env.step(action)
            
            # Updating q-values
            Q.update_values(state, action, reward, state_, done)
            
            # logging and moving to the next state
            step += 1; state = state_
            if (step-1)%print_freq == 0:       
                eval_total_reward, eval_successes, _ = evaluate(task_env, skill=Q, episodes=eval_episodes, epsilon=0, gamma=gamma, seed=seed)
                if eval_total_reward >= best_total_reward:
                    best_total_reward = eval_total_reward
                    torch.save(Q, q_dir+"skill")
                logger.record("steps", step); logger.record("episodes", num_episodes); 
                logger.record("total reward", reward_total); logger.record("successes", successes/num_episodes)
                logger.record("eval total reward", eval_total_reward); logger.record("eval successes", eval_successes)
                logger.record("time elapsed", time.time()-start_time); logger.dump(step)
                reward_total, successes, num_episodes, start_time = 0, 0, 1, time.time()
            if done or truncated: 
                num_episodes += 1; reward_total += reward; successes += reward>=task_env.rmax
                break
    return Q

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="The agent's environment, or the RM augmented environment for a predefined task distribution.", default='Office-v0')
parser.add_argument("--ltl", help="The ltl task the agent should solve. Ignored if --env is a task. E.g. (F (c & X (F o))) & (G~d)", default='')
parser.add_argument("--total_steps", help="Total training steps", type=int, default=100000)
parser.add_argument("--q_dir", help="Directory where the learned task specific skill will be saved", default='')
parser.add_argument("--log_dir", help="Directory where the results will be saved", default='')
parser.add_argument("--qinit", help="Q-values optimistic initialisation", type=float, default=0.0)
parser.add_argument("--seed", help="Random seed", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    gym.logger.set_level(gym.logger.ERROR) 
    log_dir = args.log_dir if args.log_dir else './data/logs/' + args.env + "/"
    q_dir = args.q_dir if args.q_dir else './data/ql/' + args.env + "/"
    os.makedirs(q_dir, exist_ok=True); 
    assert ("Task" in args.env) or args.ltl, "A task distribution or ltl specification must be provided"
    
    # Initialise MDP for the given temporal logic task (task_env)
    if "Task" in args.env: task_env = gym.make(args.env)
    else:                  task_env = Task(gym.make(args.env), args.ltl)

    # Train task specific skill (skill_task) fewshot
    skill_task = learn(task_env, args.total_steps, q_dir, log_dir, qinit=args.qinit, seed=args.seed)
