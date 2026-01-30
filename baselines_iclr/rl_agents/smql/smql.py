import sys, os, argparse, random, time, torch, numpy as np, gymnasium, gym, envs
sys.path.insert(1, '../')
from sm import TaskPrimitive, SkillMachine, BaseAgent
from rm import Task
from baselines import logger


class gym_nasium(gymnasium.Wrapper):
    """ Converts the gym environment from the Icarte et al Reward Machines codebase to the Gymnasium environment we expect for Skill Machines"""
    
    def __init__(self,env,task=False):
        super().__init__(env)
        self.env = env
        self.task = task
        if not self.task:
            self.observation_space = gymnasium.spaces.Box(low=env.observation_space.low.min(), high=env.observation_space.high.max(), shape=(env.observation_space.shape[0],), dtype=env.observation_space.dtype)
        else:
            self.observation_space_dict = {}
            for key,space in [('env_state',env.observation_dict["features"]),('rm_state',env.observation_dict["rm-state"])]:
                self.observation_space_dict[key] = gymnasium.spaces.Box(low=space.low.min(), high=space.high.max(), shape=(space.shape[0],), dtype=space.dtype)
            self.observation_space = gymnasium.spaces.Dict(self.observation_space_dict)
        if type(env.action_space) != gym.spaces.Box: 
            self.action_space = gymnasium.spaces.Discrete(env.action_space.n)
        else: 
            self.action_space = gymnasium.spaces.Box(low=env.action_space.low.min(), high=env.action_space.high.max(), shape=(env.action_space.shape[0],), dtype=env.action_space.dtype)
    
    def convert_rm(self):
        rm = self.env.current_rm
        self.rm = lambda: None
        self.rm.rmin, self.rm.rmax = 0, 1
        self.rm.u0 = rm.u0
        self.rm.u = rm.u0
        self.rm.terminal_states = [rm.terminal_u]
        self.rm.delta_u = {u:{exp:u_                                      for u_,exp in rm.delta_u_full[u].items()} for u in rm.delta_u_full}
        self.rm.delta_r = {u:{exp:rm.delta_r_full[u][u_].get_reward(None) for u_,exp in rm.delta_u_full[u].items()} for u in rm.delta_u_full}
            
    def reset(self, **kwargs):
        state = self.env.reset()
        info = {"true_propositions": self.env.get_predicates()}
        if not self.task: 
            if type(state) == tuple: state = np.array(state)
        else:
            self.convert_rm()
            state = {'env_state': state[:len(self.env.obs)],'rm_state': state[len(self.env.obs):]}
        return state, info
    
    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action)
        info.update({"true_propositions": self.env.get_predicates()})
        if not self.task:
            if type(state) == tuple: state = np.array(state) 
        else:
            self.rm.u = self.current_u_id
            state = {'env_state': state[:len(self.env.obs)],'rm_state': state[len(self.env.obs):]}
        return state, reward, done, False, info

class QLAgent(BaseAgent):
    """Q-learning Agent"""
    
    def __init__(self, name, env, SM=None, save_dir=None, load=False, lr=0.5, gamma=0.9, qinit=0):
        self.values, self.lr, self.gamma, self.SM, self.qinit = {}, lr, gamma, SM, qinit
        self.action_space, self.observation_space = env.action_space, env.observation_space
        if load: self.values = torch.load(save_dir+"wvf_"+name)
        self.name = name
                
    def get_action_value(self, state):
        stateb = gymnasium.spaces.flatten(self.observation_space, state).tobytes()
        if stateb not in self.values: self.values[stateb] = np.zeros(self.action_space.n)
        action, value = self.values[stateb].argmax(), self.values[stateb].max()
        if self.SM:
            (Q_action, Q_value), ((SM_action,), (SM_value,)) = (action, value), self.SM.get_action_value(state)
            idx = np.argmax([self.gamma*Q_value, (1-self.gamma)*SM_value])
            action, value = [Q_action, SM_action][idx], [Q_value, SM_value][idx]
        return np.array([action]), np.array([value])
        
    def get_values(self, state):
        state = gymnasium.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        return self.values[state]    
    
    def update_values(self, state, action, reward, state_, done):
        state = gymnasium.spaces.flatten(self.observation_space, state).tobytes()
        if state not in self.values: self.values[state] = np.zeros(self.action_space.n) + self.qinit
        if done: self.values[state][action] += self.lr * (reward - self.values[state][action])
        else:    self.values[state][action] += self.lr * (reward + self.gamma*self.get_values(state_).max() - self.values[state][action])


def learn(env, total_timesteps=100000, zeroshot=False, fewshot=False, q_dir="vf", sp_dir="wvfs", log_dir="logs", load=False, gamma=0.9, lr=0.5, epsilon=0.5, qinit=0, eval_episodes=1, init_eval=True, print_freq=10000, seed=None, **kwargs):  
    """Q-Learning based method for solving temporal logic tasks zeroshot or fewshot using Skill Machines"""
    
    os.makedirs(sp_dir, exist_ok=True); os.makedirs(q_dir, exist_ok=True)

    # Initialise MDP for the primitives (primitive_env), and MDP for the given temporal logic task (task_env)
    if zeroshot or fewshot:
        load = True
        primitive_env = TaskPrimitive(gym_nasium(env.env.env.env), max_episode_steps=1000)
        task_env = gym_nasium(env,task=True)
    else:
        primitive_env = TaskPrimitive(gym_nasium(env), max_episode_steps=1000)
        task_env = None
    
    # Initialise the World Value Functions for the min ("0") and max ("1") WVFs
    SP = {primitive: QLAgent(primitive, primitive_env, save_dir=sp_dir, load=load, lr=lr, gamma=gamma, qinit=qinit) for primitive in ['0','1']}
    if load: primitive_env.goals.update(torch.load(sp_dir+"goals")) 
    # Skill machine over the learned skill primitives. goal_directed=True gives faster runtime since goals are maximised only per rm transition (and not per step)
    SM = SkillMachine(primitive_env, SP, goal_directed=True) 
    # Initialise task specific value function for fewshot learning
    if fewshot: Q = QLAgent("skill", task_env, SM=SM, lr=lr, gamma=gamma, qinit=qinit)

    # Start Training
    learning_epsilon = epsilon
    init_eval = print_freq if init_eval else 0 
    step, reward_total, successes, best_reward_total, num_episodes, start_time = 0, 0, 0, 0, 1, time.time()
    while step < total_timesteps:
        if num_episodes % 2 == 0:
            epsilon = learning_epsilon
        else:
            epsilon = 0.1

        if fewshot or zeroshot:
            state, info = task_env.reset(seed=seed) 
            SM.reset(task_env.rm, info["true_propositions"])
        else:
            state, info = primitive_env.reset(seed=seed) 
        
        while True:  
            # Selecting and executing the action
            if fewshot or zeroshot:
                states = {k:np.expand_dims(v,0) for (k,v) in state.items()}
                if random.random() < epsilon: action = task_env.action_space.sample()
                elif fewshot:                 action = Q.get_action_value(states)[0][0]
                else:                         action = SM.get_action_value(states)[0][0]
                state_, reward, done, truncated, info = task_env.step(action)
                SM.step(task_env.rm, info["true_propositions"])
            else:
                if random.random() < epsilon: action = primitive_env.environment.action_space.sample()
                else:                         action = random.choice([action for action in range(primitive_env.action_space.n) if SP["1"].get_values(state)[action] == SP["1"].get_values(state).max()])
                state_, reward, done, truncated, info = primitive_env.step(action)
            
            # Updating q-values
            if fewshot and num_episodes % 2 == 0 and step>=init_eval: 
                Q.update_values(state, action, reward, state_, done)
            elif (not zeroshot) and num_episodes % 2 == 0 and step>=init_eval:
                tp_state, tp_state_ = state.copy(), state_.copy()
                for primitive in SP:
                    primitive_env.primitive = primitive
                    for desired_goal in primitive_env.goals.values():
                        tp_state["desired_goal"], tp_state_["desired_goal"] = desired_goal, desired_goal
                        tp_reward = primitive_env.compute_reward(info["achieved_goal"].reshape(1,-1), desired_goal.reshape(1,-1), info["env_reward"], done)[0]
                        SP[primitive].update_values(tp_state, action, tp_reward, tp_state_, done)
            
            # logging and moving to the next state
            state = state_
            if num_episodes % 2 != 0:
                reward_total += reward
                successes += (reward>=1)
            if num_episodes % 2 == 0:
                step += 1
                if step%print_freq == 0 or step == init_eval:
                    if reward_total >= best_reward_total:
                        best_reward_total = reward_total
                        # if fewshot: torch.save(Q, q_dir+"skill")
                        if not zeroshot:
                            for primitive in SP: torch.save(SP[primitive].values, sp_dir+"wvf_"+primitive)
                            torch.save(primitive_env.goals, sp_dir+"goals")
                    logger.record_tabular("steps", step)
                    logger.record_tabular("episodes", num_episodes*0.5)
                    logger.record_tabular("eval total reward", reward_total)
                    logger.record_tabular("eval successes", successes/(num_episodes*0.5))
                    logger.record_tabular("goals", len(primitive_env.goals))
                    logger.dump_tabular()
                    num_episodes, reward_total, successes = 0, 0, 0
            if done or truncated:
                num_episodes += 1 
                break
            
    return SP

