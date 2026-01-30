from abc import ABC, abstractmethod
from collections import deque, defaultdict
import gymnasium as gym, warnings, numpy as np, imageio
from gymnasium.utils import seeding
from sympy.logic import boolalg
from sympy import sympify, Symbol


class BaseAgent(ABC):
    """Basic RL Agent interface expected by this codebase"""
    @abstractmethod
    def get_action_value(self, states):
        """Agents should have a get_action function that takes as input the state dictionary and returns the action and value"""
        raise NotImplementedError


class TaskPrimitive(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    def __init__(self, env, primitive="1", rmin=0, rmax=1, max_episode_steps=100, use_env_rewards=False, wvf=True, sb3=False, seed=None, render_mode="human", **kwargs):
        """
        Environment for learning skill primitives.
        env (gym env): the environment
        primitive (string): the proposition (e.g. "p_coffee") or constraint (e.g. "c_coffee") defining the task primitive. 
                           It can also be '1' (max rewards) or '0' (min rewards) respectively for the max or min task primitives respectively.  
        rmin (int): The minimum reward of the reward machines defining temporal logic tasks in this environment
        rmax (int): The maximum reward of the reward machines defining temporal logic tasks in this environment
        wvf (bool): Use the extended MDP to learn a World Value Function.
        sb3 (bool): Use the GoalEnv interface required for Stable Baselines3's Hindsight Experience Replay (HER) agent.
                                 To easily handle the done action with Stable baselines3, we don't augment the environments action space with a done action, 
                                 and simply assume the agent always chooses the done action only when it reaches the desired goal.
        """
        assert hasattr(env, "predicates"), "Your environment does not contain the list of all atomic predicates: predicates"
        assert hasattr(env, "get_predicates"), "Your environment doas not contain the get_predicates method: get_predicates()-->true_propositions"
        if not hasattr(env, "constraints"): 
            warnings.warn("Your environment does not contain the list of all constraints. We will set constraints=predicates by default.")
            env.constraints=env.predicates
        
        self.render_mode = render_mode
        self.environment, self.primitive, self.rmin, self.rmax, self.use_env_rewards = env, primitive, rmin, rmax, use_env_rewards
        (self.np_random, self.seed), self.max_episode_steps = seeding.np_random(seed), max_episode_steps
        self.rewards, self.successes = deque(maxlen=max_episode_steps*100), deque(maxlen=max_episode_steps*100)
        self.wvf, self.sb3 = wvf or sb3, sb3
        
        self.constraints_mask = np.array([1*(e in env.constraints) for e in env.predicates], dtype=np.uint8)
        self.predicates, self.get_predicates, self.constraints = env.predicates, env.get_predicates, env.constraints
        self.goal_space = gym.spaces.Box(low=0, high=1, shape=(2*len(env.predicates),), dtype=np.uint8)
        self.goals = {self.goal_space.low.tobytes(): self.goal_space.low.copy()} # Goals buffer
        self.proposition_space = gym.spaces.Box(low=0, high=1, shape=(len(env.predicates),), dtype=np.uint8)
        
        # Adding violated constraints to env states. 'achieved_goal' and 'desired_goal' are for the GoalEnv interface
        self.observation_space_dict = {'env_state': env.observation_space, 'violated_constraints': self.proposition_space} 
        if self.wvf: self.observation_space_dict.update({'desired_goal': self.goal_space})
        if self.sb3: self.observation_space_dict.update({'achieved_goal': self.goal_space})
        self.observation_space = gym.spaces.Dict(self.observation_space_dict)
        # Adding a Done action to env actions
        if hasattr(env,"actions"): self.actions = env.actions
        self.action_space = env.action_space
        if not self.sb3:
            if type(env.action_space) != gym.spaces.Box: self.action_space = gym.spaces.Discrete(env.action_space.n*2)
            else: self.action_space = gym.spaces.Box(low=env.action_space.low.min(), high=env.action_space.high.max(), shape=(env.action_space.shape[0]+1,), dtype=env.action_space.dtype)
    
    def split_action(self, action=None):
        if self.sb3:                                    return action, False
        elif type(self.action_space) == gym.spaces.Box: return action[:-1], action[-1] >= (self.action_space.low.min()+self.action_space.low.max())/2
        else:                                           return action % self.environment.action_space.n, action >= self.environment.action_space.n
        
    def compute_reward(self, achieved_goal, desired_goal=None, env_reward=[0], done=False):
        if self.primitive[0] == "p":
            idx = self.environment.predicates.index(self.primitive[2:])
            successes = achieved_goal[:, idx] == 1
        elif self.primitive[0] == "c":
            idx = self.environment.predicates.index(self.primitive[2:])
            successes = achieved_goal[:, len(self.environment.predicates) + idx] == 1
        elif self.primitive[0] == "1":
            successes = np.ones(achieved_goal.shape[0])
        else: # self.primitive[0] == "0":
            successes = np.zeros(achieved_goal.shape[0])
        
        # Non terminal rewards
        reward = env_reward if self.use_env_rewards else np.zeros(achieved_goal.shape[0])
        # Terminal rewards
        reward = (1-done)*reward + done*(successes*self.rmax + (1-successes)*self.rmin)    
        # Extended rewards
        if self.wvf:
            extend = done*(1-(achieved_goal==desired_goal).min(1))
            reward = extend*self.rmin + (1-extend)*reward

        return reward
        
    def reset(self, seed=None, **kwargs):
        self.steps, (self.np_random, self.seed) = 0, seeding.np_random(seed)
        self.env_state, env_info = self.environment.reset(seed=seed, **kwargs)
        self.true_propositions = self.environment.get_predicates()
        self.violated_constraints = (self.proposition_space.sample() & self.constraints_mask)*self.np_random.choice([0,1])

        self.achieved_goal = np.concatenate([self.true_propositions, self.violated_constraints])
        ag_key = self.achieved_goal.tobytes()
        if ag_key not in self.goals: self.goals[ag_key] = self.achieved_goal
        self.desired_goal = self.np_random.choice(list(self.goals.values()))
        self.desired_goal_key = self.desired_goal.tobytes()
        
        info = {"true_propositions": self.true_propositions, 'violated_constraints': self.violated_constraints}
        info.update(env_info)
        
        state = {'env_state': self.env_state.copy(), 'violated_constraints': self.violated_constraints.copy()}
        if self.wvf: state.update({'desired_goal': self.desired_goal.copy()})
        if self.sb3: state.update({'achieved_goal': self.achieved_goal.copy()})
        return state, info
    
    def step(self, action):
        env_action, done_action = self.split_action(action)
        self.env_state, env_reward, env_done, env_truncated, env_info = self.environment.step(env_action)
             
        true_propositions, violated_constraints = self.true_propositions, self.violated_constraints.copy()
        self.true_propositions = self.environment.get_predicates()
        self.violated_constraints |= ((true_propositions ^ self.true_propositions) & self.constraints_mask)
        self.achieved_goal = np.concatenate([self.true_propositions, violated_constraints])
        ag_key = self.achieved_goal.tobytes()
        if ag_key not in self.goals: self.goals[ag_key] = self.achieved_goal
        
        # If sb3==True, we only choose the done_action when the goal is reached. 
        # This makes intergration with existing RL libraries like SB3 easier
        if self.sb3: done_action = (self.achieved_goal==self.desired_goal).min()
        done = env_done or done_action

        state = {'env_state': self.env_state.copy(), 'violated_constraints': self.violated_constraints.copy()}
        if self.wvf: state.update({'desired_goal': self.desired_goal.copy()})
        if self.sb3: state.update({'achieved_goal': self.achieved_goal.copy()})
                
        reward = self.compute_reward(self.achieved_goal.reshape(1,-1), self.desired_goal.reshape(1,-1), env_reward, done)[0]
        self.rewards.append(reward); self.successes.append(reward>=self.rmax)
        
        truncated, self.steps = env_truncated, self.steps+1
        if self.steps>=self.max_episode_steps: truncated = True
        info = {"true_propositions": self.true_propositions, "env_reward": env_reward, "env_done": env_done, 'achieved_goal': self.achieved_goal}
        info.update(env_info)
        
        return state, reward, done, truncated, info
    
    def render(self, *args, **kwargs):
        return self.environment.render(*args, **kwargs)
    

class SkillPrimitive(BaseAgent):
    def __init__(self, primitive, wvfs, goals, predicates):
        self.predicates, self.primitive, self.wvfs, self.goals, self.goal = predicates, primitive, wvfs, goals, None 
        self.is_discrete = type(self.wvfs["1"].action_space)==gym.spaces.Discrete
        assert ("0" in wvfs) and ("1" in wvfs), "wvfs should contain the min wvf ('0') and max wvf ('1')."
    
    def get_action_value(self, states, desired_goal=None, vectorised=False):
        states, shape = states.copy(), states["env_state"].shape
        goals = np.array(list(self.goals.values()))
        if type(desired_goal) != type(None):
            states["desired_goal"] = np.expand_dims(desired_goal, axis=0)
            return self.wvfs["1"].get_action_value(states)
        elif not vectorised:
            ### Single state
            actions, values = [], []
            for desired_goal in goals:
                states["desired_goal"] = np.expand_dims(desired_goal, axis=0)
                action, value = self.get_action_goal_values(states)
                actions.append(action[0]); values.append(value[0])
            goal_idx = np.argmax(values); self.goal = goals[goal_idx]
            if self.is_discrete: return [actions[goal_idx]], [values[goal_idx]]
            else:                return actions[goal_idx], values[goal_idx]
        else:
            ### Many states
            n_goals, states["desired_goal"] = len(goals), np.concatenate([goals]*shape[0], axis=0)
            for key in states.keys():
                if key != "desired_goal": states[key] = np.repeat(states[key], repeats=n_goals, axis=0)
            actions, values = self.get_action_goal_values(states)
            mask = [np.argmax(values[s*n_goals:(s+1)*n_goals]) + s*n_goals for s in range(shape[0])]
            if shape[0]==1: self.goal = states["desired_goal"][mask,:][0]
            if self.is_discrete: return actions[mask], values[mask]
            else:                return actions[mask,:], values[mask]
    
    def get_action_goal_values(self, states):        
        if self.primitive in self.wvfs:
            actions, values = self.wvfs[self.primitive].get_action_value(states)
        else:
            goal_prop = states["desired_goal"][:,-2*len(self.predicates):-len(self.predicates)]
            goal_cons = states["desired_goal"][:,-len(self.predicates):]
            idx = self.predicates.index(self.primitive[2:])
            if self.primitive[0] == "p": mask = goal_prop[:, idx] == 1 
            else:                        mask = goal_cons[:, idx] == 1
            max_actions, max_values = self.wvfs["1"].get_action_value(states)
            min_actions, min_values = self.wvfs["0"].get_action_value(states)
            if self.is_discrete: actions = np.where(mask, max_actions, min_actions)
            else:                actions = np.where(mask[:,np.newaxis], max_actions, min_actions)
            values = np.where(mask, max_values, min_values)
        return actions, values    

class ComposeSkillPrimitive(SkillPrimitive):
    def __init__(self, skill_primitives, compose="or"):
        self.compose = compose
        self.skill_primitives = skill_primitives    
        super().__init__(compose, skill_primitives[0].wvfs, skill_primitives[0].goals, skill_primitives[0].predicates)    
    
    def get_action_goal_values(self, states):
        all_actions, all_values = [], []
        for skill_primitive in self.skill_primitives:
            actions, values = skill_primitive.get_action_goal_values(states)
            all_actions.append(actions); all_values.append(values)
        all_actions, all_values = np.array(all_actions), np.array(all_values)

        if self.compose=="or":
            mask = np.array(np.argmax(all_values, axis=0))
            actions, values = all_actions[mask,np.arange(all_values.shape[1])], all_values[mask,np.arange(all_values.shape[1])]
        elif self.compose=="and":
            mask = np.argmin(all_values, axis=0)
            actions, values = all_actions[mask,np.arange(all_values.shape[1])], all_values[mask,np.arange(all_values.shape[1])]
        else: # self.compose=="not":
            max_actions, max_values = self.skill_primitives[0].wvfs["1"].get_action_value(states)
            min_actions, min_values = self.skill_primitives[0].wvfs["0"].get_action_value(states)
            values = (max_values + min_values) - values
            if self.is_discrete: actions = np.where((abs(values - max_values) < abs(values - min_values)), max_actions, min_actions)
            else:                actions = np.where((abs(values - max_values) < abs(values - min_values))[:,np.newaxis], max_actions, min_actions)
        
        return actions, values

def value_iteration(delta_u, delta_r, gamma=0.9):
    """
    We use value iteration to compute the Q-function for a reward machine.
    """
    Q, V = defaultdict(lambda: defaultdict(float)), defaultdict(float)
    V_error = 1
    while V_error > 0.0000001:
        V_error = 0
        for u in delta_u:
            q = []
            for exp in delta_u[u]:
                u_next, r = delta_u[u][exp], delta_r[u][exp]
                q.append(r+gamma*V[u_next])
                Q[u][exp] = q[-1]
            V_error = max([V_error, abs(max(q)-V[u])])
            V[u] = max(q)
    return Q

def exp_wvf(exp, skill_primitives):
    """
    Composes a set of skill primitives according to a given Boolean expression (exp).
    exp (string): the Boolean expression over predicates.
    skill_primitives (dict): dictionary of world value functions (WVFs). The keys are the predicates for each primitive, and the values are their corresponding WVFs.
    Returns (WVF): The composition of skill_primitives according to the Boolean expression
    """
    
    exp = sympify(exp)
    if not exp: return skill_primitives["0"]

    def convert(exp):
        if type(exp) in [boolalg.Or,boolalg.And]: compound = convert(exp.args[0])
        if type(exp) == Symbol: compound = skill_primitives[str(exp)]
        elif type(exp) == boolalg.Or:
            for sub in exp.args[1:]:   compound = ComposeSkillPrimitive([compound, convert(sub)], compose="or")
        elif type(exp) == boolalg.And:
            for sub in exp.args[1:]:   compound = ComposeSkillPrimitive([compound, convert(sub)], compose="and")
        elif type(exp) == boolalg.Not: compound = ComposeSkillPrimitive([convert(exp.args[0])], compose="not")
        elif type(exp) in [bool,boolalg.BooleanTrue,boolalg.BooleanFalse]:          
            compound = skill_primitives["1"] if exp else skill_primitives["0"]
        else: 
            assert False, "There is an unknown symbol in the expression: "+ str(exp) + str(type(exp))
        return compound   
    return convert(exp)


class SkillMachine():
    def __init__(self, primitive_env, skill_primitives, vectorised=False, goal_directed=False, rebuild_onreset=True):
        self.vectorised = vectorised # True if the learned policies can take a list of multiple states as input. E.g. When using a neural network
        self.rebuild_onreset = rebuild_onreset # Rebuild Skill Machine after each environment reset. Set to False for speed up when the tasks per episode don't change.
        self.goal_directed = goal_directed
        self.rm_state, self.graph, self.exp_wvf_cache = None, None, {}
        
        # Generate all skill primitives
        self.skill_primitives = {primitive: SkillPrimitive(primitive, skill_primitives, primitive_env.goals, primitive_env.predicates) for primitive in skill_primitives}
        for proposition in primitive_env.predicates: self.skill_primitives[("p_"+proposition)] = SkillPrimitive("p_"+proposition, skill_primitives, primitive_env.goals, primitive_env.predicates)
        for constraint in primitive_env.constraints: self.skill_primitives[("c_"+constraint)]  = SkillPrimitive("c_"+constraint,  skill_primitives, primitive_env.goals, primitive_env.predicates)  
        
        # Same spaces defined for Primitives
        self.constraints = primitive_env.constraints
        self.constraints_mask = primitive_env.constraints_mask
        self.proposition_space = primitive_env.proposition_space
        self.goal_space = primitive_env.goal_space
        self.action_space = primitive_env.environment.action_space
        self.split_action = primitive_env.split_action
       
    def build(self, rm):
        self.delta_q, Q = defaultdict(bool), value_iteration(rm.delta_u, rm.delta_r)
        self.sm_str = "" # In a format similar to that of RMs from Rodrigo's code base
        for u in rm.delta_u:
            exp_predicates, exp_constraints, best = "1", "0", float("-inf")
            for exp, value in Q[u].items():
                if value > best:     exp_predicates = exp; best = value
                if value <= rm.rmin: exp_constraints += (" | " + exp) 
            
            exp_predicates = sympify(exp_predicates.replace("!","~").replace("1","true").replace("0","false")).simplify()
            exp_constraints = sympify(exp_constraints.replace("!","~").replace("1","true").replace("0","false")).simplify()
            if type(exp_predicates)!=bool: 
                exp_predicates = exp_predicates.subs({symbol: f'p_{symbol}' for symbol in exp_predicates.free_symbols})
            if type(exp_constraints)!=bool: 
                # Only consider learned constraints
                symbols = {symbol: (f'c_{symbol}' if f'c_{symbol}' in self.skill_primitives else "false") for symbol in exp_constraints.free_symbols}
                exp_constraints = exp_constraints.subs(symbols)
            self.delta_q[u] = exp_predicates & ~ exp_constraints
            
            for exp, u_next in rm.delta_u[u].items():
                self.sm_str += "({},{},'{}',WorldValueFunction({}))\n".format(u,rm.delta_u[u][exp],exp,self.delta_q[u])
        self.sm_str = "{}    # terminal state\n".format(list(rm.terminal_states)) + self.sm_str
        self.sm_str = "{}    # initial state\n".format(rm.u0) + self.sm_str
        
    def reset(self, rm, true_propositions):
        self.rm_state, self.goal = None, None
        self.violated_constraints, self.true_propositions = self.proposition_space.low.copy(), true_propositions.copy()
        if self.rebuild_onreset: self.build(rm)
        return self.step(rm, true_propositions)
    
    def step(self, rm, true_propositions):
        if self.rm_state != rm.u:
            self.violated_constraints, self.true_propositions = self.proposition_space.low.copy(), true_propositions.copy()
            self.rm_state, self.exp = rm.u, self.delta_q[rm.u]
            if self.exp not in self.exp_wvf_cache:
                self.wvf = exp_wvf(self.exp, self.skill_primitives) 
                self.exp_wvf_cache[self.exp] = self.wvf
            self.wvf = self.exp_wvf_cache[self.exp]
            self.goal = None
        else:
            self.violated_constraints |= ((self.true_propositions ^ true_propositions) & self.constraints_mask)
            self.true_propositions = true_propositions.copy()
            if self.goal_directed: self.goal = self.wvf.goal
        return self.wvf
    
    def get_action_value(self, states):  
        states = {'env_state': states["env_state"] , 'violated_constraints': np.repeat(np.expand_dims(self.violated_constraints, 0), repeats=len(states["env_state"]), axis=0), 
                                                     'achieved_goal': np.repeat(np.expand_dims(self.goal_space.low, 0), repeats=len(states["env_state"]), axis=0)}
        action, value = self.wvf.get_action_value(states, desired_goal=self.goal, vectorised=self.vectorised)
        env_action, self.done_action = self.split_action(action)
        return env_action, value

    
def evaluate(task_env, SM=None, skill=None, epsilon=0, gamma=1, episodes=100, eval_steps=1000, seed=None):
    """Given a temporal logic task (task_env), evaluates the SM possibly combined with a task specific policy (skill)"""

    rewards, successes, steps, episode = 0, 0, 0, 0
    while episode<episodes:
        episode += 1; step = 0
        state, info = task_env.reset(seed=seed)
        if SM: SM.reset(task_env.rm, info["true_propositions"])
        while True:
            states = {k:np.expand_dims(v,0) for (k,v) in state.items()}
            if np.random.random() < epsilon: action = task_env.action_space.sample()
            elif skill:                      action = skill.get_action_value(states)[0][0]
            else:                            action = SM.get_action_value(states)[0][0]
            state, reward, done, truncated, info = task_env.step(action)       
            if SM: SM.step(task_env.rm, info["true_propositions"])
            
            rewards += (gamma**step)*reward
            successes += reward>=task_env.rm.rmax
            step += 1; steps += 1
            if done or truncated or step>=eval_steps: break
    return rewards, successes/episodes, steps/episodes