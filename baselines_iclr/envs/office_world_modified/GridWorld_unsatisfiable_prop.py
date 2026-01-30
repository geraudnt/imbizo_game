from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from reward_machines.rm_environment import RewardMachineEnv

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from collections import defaultdict
from itertools import chain, combinations
from copy import deepcopy
from string import ascii_lowercase
from collections import OrderedDict

from PIL import Image

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})

COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0.9,0.9,0.9], 10: [0, 0, 1], 20:[1, 1, 0.0], 21:[0.8, 0.8, 0.8]}


### Predicates base class

class GridWorld_Object():
    def __init__(self, positions=[], count=float('inf'), track=False):
        self.track = track
        self.positions = positions
        self.random_state = count==None
        self._count = lambda: count if count else float('inf')
        
        self.achieved = False
        self.count = {position: self._count() for position in self.positions}

    def reset(self):
        self.count = {position: self._count() for position in self.positions}
        self.achieved = False
        return None
    
    def state(self, position):
        achieved = (position in self.count)
        if achieved and self.count[position]:
            self.count[position] -= 1
            if self.random_state and np.random.random()>0.5:
                self.count[position] = 0
            achieved = True
        
        achieved = self.achieved or achieved
        if self.track:
            self.achieved = achieved
        
        state = []
        for position in self.positions:
            state.append(self.count[position]>0)
        
        return achieved, tuple(state)


### Office world objects

MAP =   "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1\n" \
        "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"

class roomA(GridWorld_Object):
    def __init__(self):
        positions = [(2,2)]
        super().__init__(positions)
        
class roomB(GridWorld_Object):
    def __init__(self):
        positions = [(2,14)]
        super().__init__(positions)
        
class roomC(GridWorld_Object):
    def __init__(self):
        positions = [(10,14)]
        super().__init__(positions)
        
class roomD(GridWorld_Object):
    def __init__(self):
        positions = [(10,2)]
        super().__init__(positions)

class coffee(GridWorld_Object):
    def __init__(self):
        positions = [(3,5),(9,11)]
        super().__init__(positions)

class mail(GridWorld_Object):
    def __init__(self, count=1):
        positions = [(6,10)]
        super().__init__(positions, count)

class office(GridWorld_Object):
    def __init__(self, count=1):
        positions = [(6,6)]
        super().__init__(positions, count)

class decor(GridWorld_Object):
    def __init__(self):
        # positions = [(6,2),(6,14),(2,6),(2,10),(10,6),(10,10),(2,2),(2,14),(10,14),(10,2)] # If room predicates are not used
        positions = [(6,2),(6,14),(2,6),(2,10),(10,6),(10,10)]
        super().__init__(positions)

gridworld_objects =  {
    '1room': roomA(),
    '2room': roomB(),
    '3room': roomC(),
    '4room': roomD(),
    'decor': decor(),
    'coffee': coffee(),
    'mail': mail(),
    'office': office(),
}


### GridWorld environment

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, MAP=MAP, gridworld_objects=gridworld_objects, step_reward=0, start_positions=None, slip_prob=0):

        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possiblePositions = []
        self.walls = []
        
        self.MAP = MAP
        self._map_init()
        self.diameter = (self.n+self.m)-4

        self.done = False
        
        self.slip_prob = slip_prob
        
        self.gridworld_objects = gridworld_objects
        self.gridworld_objects_keys =tuple(sorted(list(self.gridworld_objects.keys())))

        self.start_positions = start_positions
        self.position = self.start_positions if start_positions else (1, 1)
        
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(self.position))
        self.state = self.position,tuple(object_states)

        # Rewards
        self.step_reward = step_reward
        
        # Defining actions
        self.actions = dict(
                            UP = 0,
                            RIGHT = 1,
                            DOWN = 2,
                            LEFT = 3,
                            )
 
        # Gym spaces for observation and action space
        self.observation_space = spaces.Box(low=0, high=max([self.n,self.m]), shape=(2,), dtype=np.uint8) #spaces.Discrete(len(self.possiblePositions))
        self.action_space = spaces.Discrete(4)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def pertube_action(self,action): 
        a = 1-self.slip_prob
        b = self.slip_prob/(self.action_space.n-2)
        if action == self.actions['UP']:
            probs = [a,b,b,b]
        elif action == self.actions['RIGHT']:
            probs = [b,a,b,b]
        elif action == self.actions['DOWN']:
            probs = [b,b,a,b]
        elif action == self.actions['LEFT']:
            probs = [b,b,b,a]
        action = np.random.choice(np.arange(len(probs)), p=probs)       
        return action

    def step(self, action):
        assert self.action_space.contains(action)

        action = self.pertube_action(action)
        reward = self._get_reward(self.state, action)
        
        x, y = self.position
        if action == self.actions['UP']:
            x = x - 1
        elif action == self.actions['DOWN']:
            x = x + 1
        elif action == self.actions['RIGHT']:
            y = y + 1
        elif action == self.actions['LEFT']:
            y = y - 1
        next_position = (x, y)
            
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(next_position))
        
        is_coffee = self.gridworld_objects["coffee"].state(self.position) and self.position == self.gridworld_objects["coffee"].positions[1]
        if is_coffee:
            self.done = True
        
        if self._get_grid_value(next_position) == 0:  # new position not in walls list
            self.position = next_position
        
        self.state = (self.position,tuple(object_states))
                     
        return self.state, reward, self.done, {}

    def _get_reward(self, state, action):      
        return self.step_reward 

    def reset(self):
        self.done = False
        
        if not self.start_positions:
            idx = np.random.randint(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]  # self.start_state_coord
        else:
            self.position = self.start_positions[np.random.randint(len(self.start_positions))]
        
        for p,f in self.gridworld_objects.items():
            f.reset()
        
        object_states = []
        for i in self.gridworld_objects_keys:
            object_states.append(self.gridworld_objects[i].state(self.position))
        self.state = (self.position,tuple(object_states))
        return self.state

    def render(self, agent=True, env_map=True, fig=None, mode='rgb_array', title=None, grid=False, size=1):        
        img = self._gridmap_to_img(env_map = env_map)        
        if not fig:
            fig = plt.figure(1, figsize=(12, 8), dpi=60, facecolor='w', edgecolor='k')
            # fig, ax = plt.subplots(figsize=(12, 8), dpi=60, facecolor='w', edgecolor='k')
        
        params = {'font.size': 40}
        plt.rcParams.update(params)
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.grid(grid)
        if title:
            plt.title(title, fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, self.n, self.m, 0])

        if env_map:
            ax = fig.gca()
            for position in self.possiblePositions:
                y,x = position
                for gridworld_object, function in self.gridworld_objects.items():
                    if position in function.positions and function.count[position]>0:
                        p = gridworld_object[0].upper()
                        c = function.count[position]
                        label = "${}$".format(p) if c == float('inf') else "${}_{}$".format(p,c)
                        if gridworld_object == "1room":
                            label = "$A$"
                        if gridworld_object == "2room":
                            label = "$B$"
                        if gridworld_object == "3room":
                            label = "$C$"
                        if gridworld_object == "4room":
                            label = "$D$"
                        if gridworld_object == "decor":
                            label = r"$\scaleobj{2}{\mbox{\ding{93}}}$"
                        if gridworld_object == "coffee":
                            label = r"$\scaleobj{2}{\mbox{\Coffeecup}}$"
                        if gridworld_object == "mail":
                            label = r"$\scaleobj{2}{\mbox{\Letter}}$"
                        if gridworld_object == "office":
                            label = r"$\scaleobj{2}{\mbox{\Gentsroom}}$"
                        
                        ax.text(x+0.2, y+0.8, label, style='oblique', fontweight="bold", size=fig.get_figheight()*4)
                        break
        if agent:
            y, x = self.position
            ax.add_patch(patches.Circle((x+0.5, y+0.5), radius=0.3, fc='blue', transform=ax.transData, zorder=10))
        
        fig.canvas.draw()

        if mode == 'rgb_array':
            height, width = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            return img
        
        return fig

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.walls.append((i, j))
                # possible states
                else:
                    self.possiblePositions.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possiblePositions:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, position):
        return self.grid[position[0]][position[1]]

    def _gridmap_to_img(self, agent=False, env_map=True):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):          
                P = False
                # for gridworld_object, function in self.gridworld_objects.items():
                #     if (i,j) in function.positions and function.count[(i,j)]>0:
                #         P = True
                #         break
                is_coffee = self.gridworld_objects["coffee"].state((i, j)) and (i, j) == self.gridworld_objects["coffee"].positions[1]
                for k in range(3):
                    if agent and (i, j) == self.position:#start_positions:
                        this_value = COLOURS[10][k]
                    elif P:
                        this_value = COLOURS[3][k]
                    elif is_coffee:
                        this_value = [1,0.3,0.3][k]
                    else:
                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img


### Defining tasks over the environment

predicates =  {
    '_1room': lambda state: state[1][0][0],
    '_2room': lambda state: state[1][1][0],
    '_3room': lambda state: state[1][2][0],
    '_4room': lambda state: state[1][3][0],
    '_decor': lambda state: state[1][5][0],
    '_coffee': lambda state: state[1][4][0],
    '_mail': lambda state: state[1][6][0],
    '_office': lambda state: state[1][7][0],
    
    # 't_coffee': lambda state: True in state[1][4][1],
    't_mail': lambda state: True in state[1][6][1] and state[1][6][0],
    't_office': lambda state: True in state[1][7][1] and state[1][7][0],
}

class Task(gym.core.Wrapper):
    def __init__(self, env, predicates=predicates, task_goals=[], rmax=1, rmin=0, start_positions=None):
        super().__init__(env)

        self.start_positions = start_positions
        self.rmax = rmax
        self.rmin = rmin
        self.state = None

        # All events
        self.events = ascii_lowercase[:len(predicates.keys())] #self.env.features
        self.predicates = list(self.events)
        self.constraints = ["e"] # self.events #
        self.rmax = 1
        self.rmin = 0
        self.predicate_letters = list(zip(predicates.keys(), ascii_lowercase))

        self.observation_space = spaces.Box(low=0, high=max([self.n,self.m]), shape=(2,), dtype=np.uint8)

    def get_predicates(self):
        predicates_ = np.zeros(len(self.predicates), dtype=self.observation_space.dtype)
        i = 0
        for (key, letter) in self.predicate_letters:
            if letter in self.events:
                if predicates[key](self.state) == True: predicates_[i] = 1
                i += 1
        return predicates_

    def get_events(self):
        gridworld_events = []
        for key, letter in self.predicate_letters:
            if letter in self.events and predicates[key](self.state) == True:
                gridworld_events.append(letter)
        return str().join(gridworld_events)

    def reset(self):
        self.state = self.env.reset()
        return self.state[0]

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.state = state
        return self.state[0], self.rmin, False, {}

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


class GridWorldEnv(Task):
    def __init__(self):
        super().__init__(GridWorld())

class GridWorldRMEnv(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t%d.txt"%i for i in range(1,5)]
        env = Task(GridWorld())
        super().__init__(env, rm_files)

class GridWorldRM1Env(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t1.txt"]
        env = Task(GridWorld())
        super().__init__(env, rm_files)

class GridWorldRM2Env(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t2.txt"]
        env = Task(GridWorld())
        super().__init__(env, rm_files)

class GridWorldRM3Env(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t3.txt"]
        env = Task(GridWorld())
        super().__init__(env, rm_files)


class GridWorldRM4Env(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t4.txt"]
        env = Task(GridWorld())
        super().__init__(env, rm_files)

class GridWorldRM5Env(RewardMachineEnv):
    def __init__(self): #(self, env, rm_files):
        rm_files = ["./envs/office_world_modified/reward_machines/t5.txt"]
        env = Task(GridWorld())
        super().__init__(env, rm_files)
