"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.
However, each environment must implement the following function:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
Notes:
    - The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
    - The agent only gets the reward given by the RM.
    - Rewards coming from the environment are ignored.
"""

import gym
from gym import spaces
import numpy as np
from itertools import chain, combinations, product
from reward_machines.rm_environment import RewardMachineEnv


class SPWrapper(gym.Wrapper):
    """
    SP wrapper
    --------------------
    It augments the SPs observations with the task primitive, goal, and cross product states. 
    """

    def __init__(self, env, use_spe=True, tabular=False):
        super().__init__(env)
        
        self.max_steps = float("inf")

        self.tabular = tabular
        self.use_spe = use_spe
        self.num_events = len(self.env.events)
        self.num_constraints = len(self.env.constraints)
        
        self.all_events = list(chain(*map(lambda x: combinations(list(env.events), x), range(0, self.num_events+1))))
        self.all_events = ["".join(events) for events in self.all_events] 

        self.sp_tasks = list(env.events)+["max"]
        self.num_sp_tasks = len(self.sp_tasks)

        self.sp_states = list(chain(*map(lambda x: combinations(list(env.constraints), x), range(0, self.num_constraints+1))))
        self.sp_states = ["".join(sp_state) for sp_state in self.sp_states]
        self.num_sp_states = len(self.sp_states)

        self.sp_goals = [self.add_events(e, sp) for e in env.events for sp in self.sp_states]
        self.num_sp_goals = len(self.sp_goals)

        # The observation space is a dictionary including the env features, a binary representation of the events achieved, and a binary representation of the goal to acheive
        
        # Computing binary encodings for the goal states
        # self.sp_features = {}
        # self.len_sp_features = len(self.sp_tasks)*len(self.sp_states)*len(self.all_events)
        # for i, (sp_task,sp_state,sp_goal) in enumerate(product(self.sp_tasks, self.sp_states, self.all_events)):
        #     feature = np.zeros(self.len_sp_features)
        #     feature[i] = 1
        #     self.sp_features[(sp_task,sp_state,sp_goal)] = feature
        # self.sp_features = {}
        # self.all_events += ["max"]
        # self.len_sp_features = len(self.all_events)
        # for i, sp_event in enumerate(self.all_events):
        #     feature = np.zeros(len(self.all_events))
        #     feature[self.all_events.index(sp_event)] = 1
        #     self.sp_features[sp_event] = feature
        # Computing binary encodings for the goal states
        self.len_sp_features = self.num_events
        ar = 2**np.arange(self.num_events-1, -1, -1)
        bin_int = lambda x: int((ar*x).sum())
        self.sp_features = {}
        for events in self.all_events:
            e_features = np.zeros(self.num_events)
            for e in events:
                e_id = self.events.index(e)
                e_features[e_id] = 1
            self.sp_features[events] = e_features
        self.sp_features["max"] = self.sp_features[self.events]

        self.sp_goal_seq = None
        # self.sp_goal_seq = ["b","a","c",""] # None # 
        if not use_spe:
            env = env.env.env
        else:
            # self.observation_dict  = spaces.Dict({'features': env.observation_space,
            #                                     'sp_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8)})
            self.observation_dict  = spaces.Dict({'features': env.observation_space,
                                              'sp_state_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8),
                                              'sp_task_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8),
                                              'sp_goal_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8)})
            flatdim = gym.spaces.flatdim(self.observation_dict)
            s_low  = float(env.observation_space.low[0])
            s_high = float(env.observation_space.high[0])
            self.observation_space = spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)
        
        # self.sp_observation_dict  = spaces.Dict({'features': env.observation_space,
        #                                       'sp_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8)})
        self.sp_observation_dict  = spaces.Dict({'features': env.observation_space,
                                              'sp_state_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8),
                                              'sp_task_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8),
                                              'sp_goal_features': spaces.Box(low=0, high=1, shape=(self.len_sp_features,), dtype=np.uint8)})
        flatdim = gym.spaces.flatdim(self.sp_observation_dict)
        s_low  = float(env.observation_space.low[0])
        s_high = float(env.observation_space.high[0])
        self.sp_observation_space = spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)
        print("FEATURES: ", self.len_sp_features, "ENV_SPACE", env.observation_space, "OBS_SPACE: ", self.sp_observation_space.shape, "_______________________________________________________")
        
    def reset(self):
        # Reseting the environment and updating the acheived events
        obs = self.env.reset()
        self.sp_obs = obs
        self.sp_state = self.add_events(np.random.choice(self.sp_states), self.get_constraints())
        self.sp_goal = self.add_events(np.random.choice(self.sp_goals), self.sp_state) # 

        # self.sp_goal_seq = ["abc"]
        if self.sp_goal_seq:
            # self.env.unwrapped.goal_button = 3
            # obs = self.env.reset()
            # self.sp_obs = obs
            self.sp_state = ""
            self.sp_goal = self.sp_goal_seq.pop()
        
        while not self.sp_goal:
            self.sp_goal = self.add_events(np.random.choice(self.sp_goals), self.sp_state)
        self.sp_task = "max"
        self.steps = 0
            
        self.goal_obs = obs*0
        self.sp_reward = 0
        self.info = {}

        if self.use_spe:
            obs = self.get_sp_obs(obs, self.sp_state, self.sp_goal, self.sp_task, False)
        return obs

    def step(self, action):
        # print(self.sp_goal, self.sp_state)
        events = self.get_events()
        done_state = self.add_events(events, self.sp_state)
        self.sp_reward = 1 if (done_state == self.sp_goal) else 0
        self.steps += 1
        
        if done_state not in self.sp_goals:
            self.sp_goals.append(done_state)
        if self.use_spe and done_state == self.sp_goal:
            if self.sp_goal_seq:
                if self.sp_goal == "bc":
                    self.env.unwrapped.goal_button = 0
                self.sp_state = ""
                self.sp_goal = self.sp_goal_seq.pop()
            else:
                obs = done_state  
                self.sp_obs = obs 
                reward = self.rmax if (self.sp_task in obs or self.sp_task == "max") else self.rmin     
                done = True 
                obs = self.get_sp_obs(obs, self.sp_state, self.sp_goal, self.sp_task, done)
                return obs, reward, done, self.info
        
        # executing the action in the environment
        obs, reward, done, self.info = self.env.step(action)
        self.sp_obs = obs
        self.sp_state = self.add_events(self.sp_state, self.get_constraints()) 
        if self.use_spe:
            obs = self.get_sp_obs(obs, self.sp_state, self.sp_goal, self.sp_task, done)
            for e in self.sp_goal:
                if ("reward_"+e) in self.info:
                    reward += self.info["reward_"+e]

        return obs, reward, done or self.steps>=self.max_steps, self.info
        # return obs, reward, done, self.info

    def add_events(self, events1, events2):
        events_ = events1 + events2
        events = ""
        for e in self.env.events:
            if e in events_:
                events += e
        return events

    def get_sp_obs(self, obs, sp_state, sp_goal, sp_task, done):
        if done:
            obs, sp_state = self.goal_obs, sp_state
        
        if not self.tabular:
            # sp_obs = {'features': obs,'sp_features': self.sp_features[(sp_task,sp_state,sp_goal)]}
            sp_obs = {'features': obs,'sp_state_features': self.sp_features[sp_state]
                                     ,'sp_task_features': self.sp_features[sp_task]
                                     ,'sp_goal_features': self.sp_features[sp_goal]
                     }
            flat_obs = gym.spaces.flatten(self.sp_observation_dict, sp_obs)
        else:    
            flat_obs = tuple(obs), sp_state, sp_goal, sp_task
        
        return flat_obs

    def get_experiences(self, samples, batch=None):      
        sp_states = self.sp_states if not batch else np.random.choice(self.sp_states, batch)
        sp_goals = self.sp_goals if not batch else np.random.choice(self.sp_goals, batch)
        task = "max"

        experiences = []
        for sp_state in sp_states:
            for (s,e,sps,a,r,sn,en,spsn,d) in samples:  
                sps, spsn = self.add_events(sp_state,sps), self.add_events(sp_state,spsn)
                r_ = self.env.rmin if (d and not (task in sn or task == "max")) else r
                done_state = self.add_events(e,sps)
                if done_state not in self.sp_goals:
                    self.sp_goals.append(done_state)
                done_reward = self.env.rmax if (task in done_state or task == "max") else self.env.rmin
                for goal in sp_goals:
                    if done_state != goal:
                        experience = [(s,sps,a,r_,sn,spsn,d)]
                    else:
                        experience = [(s, sps, a, done_reward, done_state, spsn, True)]
                    for obs,sp_state,action,reward,obs_next,sp_state_next,done in experience:
                        sp_obs = self.get_sp_obs(obs, sp_state, goal, task, False)
                        sp_obs_next = self.get_sp_obs(obs_next, sp_state_next, goal, task, done)
                        # experiences += [(sp_obs,action,reward,sp_obs_next,done)]  
                        yield (sp_obs,action,reward,sp_obs_next,done) 
        # return experiences


class SMWrapper(gym.Wrapper):
    """
    SM wrapper
    --------------------
    It extracts skills for each node on the RMs. 
    """

    def __init__(self, env, use_csm, sm_gamma):
        super().__init__(env)

        self.use_csm = use_csm
        self.skill_machines = []
        for rm in env.reward_machines:
            sm = self.get_skill_machine(rm, sm_gamma)
            self.skill_machines.append(sm)

    def get_skill_machine(self, rm, sm_gamma):
        q_values = value_iteration(rm.U, rm.delta_u, rm.delta_r, rm.terminal_u, sm_gamma)
        skill_machine = {}
        for u1 in q_values:
            best = float("-inf")
            for exp, value in q_values[u1].items():
                if value > best:
                    best = value
                    skill_machine[u1] = exp.replace("!","~")
        return skill_machine

def value_iteration(U, delta_u, delta_r, terminal_u, gamma):
    """
    Standard value iteration approach. 
    We use it to compute the Q-function for the RM to SM
    """
    Q = dict()
    V = dict([(u,0) for u in U])
    V[terminal_u] = 0
    V_error = 1
    while V_error > 0.0000001:
        V_error = 0
        for u1 in U:
            q_u2 = []
            Q[u1] = {}
            for u2 in delta_u[u1]:
                if delta_r[u1][u2].get_type() == "constant": 
                    r = delta_r[u1][u2].get_reward(None)
                else:
                    r = 0 # If the reward function is not constant, we assume it returns a reward of zero
                q_u2.append(r+gamma*V[u2])
                Q[u1][delta_u[u1][u2]] = q_u2[-1]
            v_new = max(q_u2)
            V_error = max([V_error, abs(v_new-V[u1])])
            V[u1] = v_new
    return Q
