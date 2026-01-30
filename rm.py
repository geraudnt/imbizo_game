"""
Reward machine from LTL task specification 
"""

import gymnasium as gym, spot, imageio, graphviz, string, re, warnings, numpy as np
from io import BytesIO
from PIL import Image
from string import ascii_lowercase
from sympy.logic import boolalg
from sympy import sympify, Symbol


def exp_boolean(exp, propositions):
    """
    Evaluates a Boolean expression. exp is the Boolean expression and propositions is a dictionary of truth values
    """
    exp = sympify(exp.replace("!","~").replace("1","True").replace("0","False"))
    if not exp: return False   

    def convert(exp):
        if type(exp) in [boolalg.Or,boolalg.And]: compound = convert(exp.args[0])

        if type(exp) == Symbol:        compound = propositions[str(exp)]
        elif type(exp) == boolalg.Or:                
            for sub in exp.args[1:]:   compound = compound or convert(sub)
        elif type(exp) == boolalg.And:
            for sub in exp.args[1:]:   compound = compound and convert(sub)
        elif type(exp) == boolalg.Not: compound = not convert(exp.args[0])
        elif type(exp) == bool:        compound = exp
        else: assert False, "There is an unknown symbol in the expression: "+ str(exp)
        return compound   
    return convert(exp)


class RewardMachine(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    def __init__(self, ltl=None, hoa=None, propositions=None, accept_terminal=True, sink_terminal=True, max_states=None, step_reward=0, rmin=0, rmax=1, seed=None, render_mode = 'human'):
        """
        Specify the reward machine using ltl or hao.
        ltl (string): The language specification in Linear Temporal Logic (LTL) format. E.g (F a) & (G~d)
        hoa (file path): The Buchi automaton specification in Hanoi Omega-Automata (HOA) format. E.g
            HOA: v1
            name: "Fa & G!d"
            States: 3
            Start: 1
            AP: 2 "a" "d"
            acc-name: Buchi
            Acceptance: 1 Inf(0)
            properties: trans-labels explicit-labels state-acc complete
            properties: deterministic stutter-invariant very-weak
            --BODY--
            State: 0 {0}
            [!1] 0
            [1] 2
            State: 1
            [0&!1] 0
            [!0&!1] 1
            [1] 2
            State: 2
            [t] 2
            --END--
        propositions (list): Sorted list of all the atomic propositions over which RMs are defined. E.g. ["a","b","d"] 
        accept_terminal (bool): Terminate when transitioning into an accepting state
        sink_terminal (bool): Terminate when transitioning into a sink/absorbing state
        max_states (int): Maximum number of RM states. Usefull to keep the observation space the same when using random ltl expressions
        step_reward (int): Reward for non-accepting/non-failure/non-absorbing transitions
        rmin (int): Reward for accepting transitions
        rmax (int): Reward for failure transitions
        """
        
        self.render_mode = render_mode
        self.render_params = dict(format="png", orientation="LR", figsize="(8,8)", dpi='200', fontsize="14", arrowsize="1", state_seperation="0.5", filename = ".automaton")
        self.propositions = propositions if propositions else ascii_lowercase[:10]
        self.n_props = len(self.propositions)
        self.seed = seed if seed else np.random.randint(0,1000000)

        ltl = ltl if ltl else spot.randltl(list(self.propositions), seed=self.seed).__next__()
        if hoa: self.automaton = spot.automaton(hoa)
        else:   self.automaton = spot.translate(spot.formula(ltl), "Buchi", "Deterministic", "Complete", "Unambiguous", "StateBasedAcceptance")
        
        assert rmin<rmax, "rmin should be stricly less than rmax"
        self.step_reward, self.rmin, self.rmax = step_reward, rmin, rmax
        self.accept_terminal, self.sink_terminal = accept_terminal, sink_terminal
        self.u, self.delta_u_cached, self.rm_str = None, {}, ""
        self.graph = None
        self.build()
        for ap in self.automaton.ap(): assert str(ap) in self.propositions, "'{0}' in the rm specification is not in AP={1}".format(str(ap), self.propositions)

        if max_states: self.observation_space = gym.spaces.Box(low=0, high=1, shape=(max_states,), dtype=np.uint8)
        else:          self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.automaton.num_states(),), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_props,), dtype=np.uint8,seed=seed)
    
    def get_reward(self, u, u_next):
        reward = self.step_reward
        if u_next in self.accept_states:
            reward = self.rmax if u!=u_next else reward
        elif u_next in self.terminal_states: 
            reward = self.rmin 
        if u==u_next and u_next in self.terminal_states: 
            reward = 0

        return reward
    
    def reset(self, seed=None, **kwargs):
        self.u = self.u0
        state = self.observation_space.low.copy()
        state[self.u] = 1
        return state, {}
    
    def step(self, action):
        action = tuple(action)
        if action in self.delta_u_cached[self.u]:
            exp = self.delta_u_cached[self.u][action]
        else:
            propositions = {self.propositions[i]:action[i] for i in range(self.n_props)}
            for exp in self.delta_u[self.u].keys():
                if exp_boolean(exp, propositions):
                    self.delta_u_cached[self.u][action] = exp
                    break
        
        u_next = self.delta_u[self.u][exp]
        reward = self.delta_r[self.u][exp]
        done = u_next in self.terminal_states
        self.u = u_next
        state = self.observation_space.low.copy()
        state[self.u] = 1
        return state, reward, done, False, {}
    
    def build(self):
        bdict = self.automaton.get_dict()
        self.u0 = self.automaton.get_init_state_number()
        transitions, self.accept_states, self.terminal_states, self.delta_u, self.delta_r = [], set(), set(), {}, {}
        for u in range(0, self.automaton.num_states()):
            U_next = list(self.automaton.out(u))
            transitions += U_next
            self.delta_u[u], self.delta_r[u], self.delta_u_cached[u] = {}, {}, {}
            if self.automaton.state_is_accepting(u): self.accept_states.add(u)
            if (len(U_next) < 1 and self.sink_terminal) \
            or (len(U_next) == 1 and U_next[0]==u and self.sink_terminal) \
            or (u in self.accept_states and self.accept_terminal): 
                self.terminal_states.add(u)
        for t in transitions:
            u, u_next, exp = t.src, t.dst, spot.bdd_format_formula(bdict, t.cond)
            reward = self.get_reward(u, u_next)
            if exp in self.delta_u[u]: continue; # Ensures deterministic FSA
            self.delta_u[u][exp], self.delta_r[u][exp] = u_next, reward

            # The reward machine in the format expected by Rodrigo's code base
            self.rm_str += "({},{},'{}',ConstantRewardFunction({}))\n".format(u,u_next,exp,reward)
        self.rm_str = "{}    # terminal state\n".format(list(self.terminal_states)) + self.rm_str
        self.rm_str = "{}    # initial state\n".format(self.u0) + self.rm_str
        
    def render(self, *args, **kwargs):
        if not self.graph:
            assert self.u!=None, "Reset the environment after instatiation before rendering"
            self.graph = graphviz.Digraph('fsm')
            self.graph.attr(rankdir=self.render_params["orientation"], size=self.render_params["figsize"], dpi=self.render_params["dpi"], ranksep=self.render_params["state_seperation"], filename = self.render_params["filename"])
            for u in self.delta_u:
                shape = 'doublecircle' if self.automaton.state_is_accepting(u) else 'circle'
                self.graph.node('u{}'.format(u), shape=shape, style='filled', fillcolor='white', fontsize=self.render_params["fontsize"])
                for exp in self.delta_u[u]:
                    u_next, reward = self.delta_u[u][exp], self.delta_r[u][exp]
                    self.graph.edge('u{}'.format(u), 'u{}'.format(u_next), label="({},{})".format(exp,reward), fontsize=self.render_params["fontsize"], arrowsize=self.render_params["arrowsize"])
            self.graph.node("start", shape="point", fontsize=self.render_params["fontsize"])
            self.graph.edge('start', 'u{}'.format(self.u0), fontsize=self.render_params["fontsize"], arrowsize=self.render_params["arrowsize"])

        image = None
        self.graph.node('u{}'.format(self.u), style='filled', fillcolor='#aaaaff')
        if self.render_mode=="human":
            self.graph.view()
        else:
            image = self.graph.pipe(self.render_params["format"])
            image = imageio.v2.imread(BytesIO(image))
        self.graph.node('u{}'.format(self.u), style='filled', fillcolor='white')
        
        return image


class Tokenizer(object):
    """Simple mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_text_size=100):
        self.vocab = {c:i for i,c in enumerate(string.printable)}
        self.dtype = np.int16
        self.max_text_size = max_text_size
        self.text_space = gym.spaces.Box(low=0, high=self.max_text_size, shape=(self.max_text_size,), dtype=self.dtype)
    
    def tokenize(self, text):
        text = text[:self.max_text_size]
        tokens = re.findall("[{0}]".format(string.printable), text.lower())
        var_indexed_text = np.array([self.vocab[token] for token in tokens])
        tokenized = np.zeros(self.max_text_size, dtype=self.dtype)
        tokenized[:len(var_indexed_text)] = var_indexed_text
        return tokenized


class Task(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    def __init__(self, env, ltl="", hoa="", tokenizer=Tokenizer, accept_terminal=True, sink_terminal=True, max_states=None, step_reward=0, rmin=0, rmax=1, seed=None, **kwargs):
        """ 
        env: The agent's environment; task: Specifies the task using ltl or a Buchi automaton in the HOA format (leave empty for a random ltl task)
        """
        assert hasattr(env, "predicates"), "Your environment does not contain the list of all atomic predicates: predicates"
        assert hasattr(env, "get_predicates"), "Your environment does not contain the get_predicates method: get_predicates()-->true_propositions"
        if not hasattr(env, "constraints"): 
            warnings.warn("Your environment does not contain the list of all constraints. We will set constraints=predicates by default.")
            env.constraints=env.predicates

        self.environment = env
        self.seed = seed
        self.tokenizer = tokenizer()
        self.task = ltl if ltl else hoa
        self.rmin, self.rmax = rmin, rmax
        self.rm_image_cache = None
        self.render_mode = env.render_mode
        self.get_predicates = env.get_predicates
        self.predicates, self.constraints = env.predicates, env.constraints
        self.rm = RewardMachine(ltl=ltl, hoa=hoa, propositions=env.predicates, accept_terminal=accept_terminal, sink_terminal=sink_terminal, max_states=max_states, step_reward=step_reward, rmin=rmin, rmax=rmax, render_mode="rgb_array")
        
        self.observation_space  = gym.spaces.Dict({'env_state': env.observation_space, 'rm_state': self.rm.observation_space, 'task': self.tokenizer.text_space})
        if hasattr(env,"actions"): self.actions = env.actions
        self.action_space = self.environment.action_space
        
    def reset(self, seed=None, **kwargs):
        (self.env_state, env_info), (rm_state, rm_info) = self.environment.reset(seed=seed, **kwargs), self.rm.reset(seed=seed, **kwargs)
        self.true_propositions = self.environment.get_predicates()
        ### Try to make terminal initial states non-terminal
        _, _, rm_done, _, _ = self.rm.step(self.true_propositions)
        tries = 0
        while rm_done: 
            (self.env_state, env_info), (rm_state, rm_info) = self.environment.reset(seed=seed, **kwargs), self.rm.reset(seed=seed, **kwargs)
            self.true_propositions = self.environment.get_predicates()
            _, _, rm_done, _, _ = self.rm.step(self.true_propositions)
            tries += 1; seed = seed+1 if type(seed)!=type(None) else seed
            if tries > 100: print("WARNING: Unable to reset the environment in a non-terminal state after 100 tries. This could be a result of the task=({0}) or seed={1}".format(self.task,seed))
        (rm_state, rm_info) = self.rm.reset(**kwargs)        
        ###
        state = {'env_state': self.env_state,'rm_state': rm_state, 'task': self.tokenizer.tokenize(self.task)}
        info = {"true_propositions": self.true_propositions, "task": self.task}; info.update(env_info); info.update(rm_info)
        return state, info
    
    def step(self, action):
        self.env_state, env_reward, env_done, env_truncated, env_info = self.environment.step(action)
        self.true_propositions = self.environment.get_predicates()
        rm_state, rm_reward, rm_done, rm_truncated, rm_info = self.rm.step(self.true_propositions)
        
        self.state = {'env_state': self.env_state,'rm_state': rm_state, 'task': self.tokenizer.tokenize(self.task)}
        reward = env_reward + rm_reward if not rm_done else rm_reward
        done = env_done or rm_done
        truncated = env_truncated or rm_truncated
        info = {"true_propositions": self.true_propositions, "task": self.task}; info.update(env_info); info.update(rm_info)

        return self.state, reward, done, truncated, info

    def render(self, *args, **kwargs):
        env_image = self.environment.render(*args, **kwargs)
        if self.render_mode=="rgb_array":
            if not self.rm_image_cache or self.rm_image_cache[0] != (self.rm.rm_str,self.rm.u): # caching to speedup rendering
                rm_image = self.rm.render()
                width, height = env_image.shape[1], int(rm_image.shape[0]*(env_image.shape[1]/rm_image.shape[1]))
                rm_image = np.array(Image.fromarray(rm_image).resize((width, height)))[:,:,:3]
                self.rm_image_cache = (self.rm.rm_str,self.rm.u), rm_image
            rm_image = self.rm_image_cache[1]
            image = np.concatenate([env_image, rm_image], axis=0)
            return image
