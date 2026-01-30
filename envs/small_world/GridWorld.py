import textwrap
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})

COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0]}

class DynamicObj():
    def __init__(self, object, max_count=1):
        self.object, self.state, self.max_count, self.count = object, object, max_count, max_count    
    def update(self):
        self.state = self.object if self.count>0 else ""     
        self.count -= 1 # bool(np.random.randint(2))  
    def reset(self):
        self.state, self.count = self.object, self.max_count  
    

class GridWorldEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    MAP = "1 1 1 1 1\n" \
          "1 0 0 0 1\n" \
          "1 0 0 0 1\n" \
          "1 0 0 0 1\n" \
          "1 1 1 1 1"

    position_predicate =  {
        (1,1):['A','B'], (2,3):['A'], (3,2):['B'], 
    }
    predicate_latex = {
        "A": "${\mbox{\Letter}}$", "B": "${\mbox{\Coffeecup}}$",
    }

    def __init__(self, MAP=MAP, start_position=(3, 3), seed=None, render_mode = 'human'):   
        self.MAP, self.start_position, self.render_mode = MAP, start_position, render_mode
        self.np_random, self.seed = seeding.np_random(seed)
        self.position = self.start_position if start_position else (1, 1)
        self.render_params = dict(dpi=100, agent=True, env_map=True, skill=None, state=None, title=None, task_title=None, skill_title=None)
        self.skill_fig, self.env_fig = None, None
        
        self.hallwayStates = None
        self.possiblePositions, self.walls = [], []
        self.n, self.m, self.grid = None, None, None
        self._map_init()
        self.map_img = self._gridmap_to_img() 
        self.skill_image = np.zeros((self.m,self.n))+float("-inf")
        self.skill_title = ""
        
        self.predicates = sorted(list(self.predicate_latex.keys()))
        self.constraints = self.predicates

        # Gym spaces for observation and action space
        self.observation_space = spaces.Box(low=0, high=max([self.n-1,self.m-1]), shape=(2,), dtype=np.uint8)
        # self.actions = dict(up = 0, right = 1, down = 2, left = 3)
        self.actions = dict(up = 0, right = 1, down = 2, left = 3, enter = 4)
        self.action_space = spaces.Discrete(len(self.actions),seed=seed)
    
    def get_predicates(self):
        predicates = np.zeros(len(self.predicates), dtype=self.observation_space.dtype)
        if self.position in self.position_predicate:
            for obj in self.position_predicate[self.position]:
                predicate = obj if type(obj)==str else obj.state
                if predicate: predicates[self.predicates.index(predicate)] = 1
        return predicates

    def step(self, action):
        assert self.action_space.contains(action)        
        x, y = self.position
        if action == self.actions['up']: x = x - 1
        elif action == self.actions['right']: y = y + 1
        elif action == self.actions['down']: x = x + 1
        elif action == self.actions['left']: y = y - 1
        next_position = (x, y)            
        if self.grid[x][y] == 0: self.position = next_position # if not wall
        state = np.array(self.position, dtype=self.observation_space.dtype)    

        if self.position in self.position_predicate:
            for obj in self.position_predicate[self.position]:
                if type(obj)!=str: obj.update()
                     
        return state, -1, False, False, {}
    
    def reset(self, seed=None, **kwargs):    
        self.np_random, self.seed = seeding.np_random(seed)

        self.position = self.start_position
        if not self.start_position:
            idx = self.np_random.integers(len(self.possiblePositions))
            self.position = self.possiblePositions[idx]
        if self.position in self.position_predicate:
            for obj in self.position_predicate[self.position]:
                if type(obj)!=str: obj.reset()
        state = np.array(self.position, dtype=self.observation_space.dtype)

        return state, {}

    def render(self, *args, **kwargs):    
        if self.render_params["skill"]: return self.render_skill(self.render_params["skill"], self.render_params["state"], task_title=self.render_params["task_title"], skill_title=self.render_params["skill_title"], dpi=self.render_params["dpi"])
        else:                           return self.render_env(render_mode=self.render_mode, task_title=self.render_params["task_title"], skill_title=self.render_params["skill_title"], dpi=self.render_params["dpi"])

    def render_env(self, render_mode="human", task_title="", skill_title="", values=None, dpi=100):    
        if not self.env_fig: self.env_fig = plt.figure("GridWorld", figsize=(12, 8), dpi=dpi, facecolor='w', edgecolor='k')
        plt.clf(); plt.xticks([]); plt.yticks([]); plt.grid(False); plt.rcParams.update({'font.size': 15})
        
        if True: # render_mode == 'hard_rgb_array': 
            # plt.imshow(self.map_img, origin="upper", extent=[0, 1, 1, 0])       
            ax = self.env_fig.gca()
            ax.set_aspect("equal", adjustable="box")  
            # colors = "red", "green", "blue"
            # color = colors[self.position[1]-1]
            # if self.position[0] == 1:
            #     ax.add_patch(patches.RegularPolygon((0.5, 0.5), numVertices=3, radius=0.46, orientation=np.pi/2, 
            #                                                          lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
            # if self.position[0] == 2:
            #     ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
            # if self.position[0] == 3:
            #     ax.add_patch(patches.Circle((0.5, 0.5), radius=0.4, lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
            colors = "red", "green", "blue"
            shapes = "triangle", "square", "circle"
            position_object = {(1,1): (colors[1],shapes[2]), (1,2): (colors[2],shapes[2]), (1,3): (colors[0],shapes[0]),
                               (2,1): (colors[2],shapes[0]), (2,2): (colors[1],shapes[0]), (2,3): (colors[2],shapes[1]),
                               (3,1): (colors[0],shapes[1]), (3,2): (colors[1],shapes[1]), (3,3): (colors[0],shapes[2])}
            color, shape = position_object[self.position]
            if shape=="triangle":
                ax.add_patch(patches.RegularPolygon((0.5, 0.5), numVertices=3, radius=0.46, orientation=np.pi/2, 
                                                                     lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
            if shape=="square":
                ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
            if shape=="circle":
                ax.add_patch(patches.Circle((0.5, 0.5), radius=0.4, lw=1, ec='black', fc=color, transform=ax.transData, zorder=10))
        else:
            plt.imshow(self.map_img, origin="upper", extent=[0, self.n, self.m, 0])        
            ax = self.env_fig.gca()
            
            if self.render_params["env_map"]:
                for pos, objects in self.position_predicate.items():
                    for obj in objects:
                        predicate = obj if type(obj)==str else obj.state
                        label = self.predicate_latex[predicate] if predicate in self.predicate_latex else ""
                        ax.text(pos[1]+0.2, pos[0]+0.8, label, style='oblique', fontweight="bold", size=self.env_fig.get_figheight()*12)        
            if self.render_params["agent"]:
                ax.add_patch(patches.Circle((self.position[1]+0.5, self.position[0]+0.5), radius=0.4, lw=0.2, ec='white', fc='black', transform=ax.transData, zorder=10)) 
         
        if task_title:
            plt.title(r" \\ ".join(textwrap.wrap(task_title, width=150)) + r" \\~\\ " + r" \\ ".join(textwrap.wrap(skill_title, width=150)), pad=20)
            plt.subplots_adjust(left=0, right=1, top=0.8, bottom=0, wspace=0, hspace=0)
        else:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        if type(values) != type(None):
            c = plt.imshow(values, cmap="RdYlBu_r", origin="upper", extent=[0, self.n, self.m, 0])  
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(c, cax=cax)
        
        plt.tight_layout()    
        if render_mode == 'rgb_array' or render_mode == 'hard_rgb_array':
            self.env_fig.canvas.draw()
            height, width = self.env_fig.get_size_inches() * self.env_fig.get_dpi()
            img = np.frombuffer(self.env_fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            plt.close(self.env_fig); self.env_fig = None
            return img
        else:
            plt.pause(0.001) 

    def render_skill(self, skill, state, task_title="", skill_title="", dpi=100):  
        if self.skill_title == skill_title:
            values = self.skill_image
        else:
            values = np.zeros((self.m,self.n))+float("-inf")
            for y, x in self.possiblePositions:
                state["env_state"][0] = np.array([y,x], dtype=np.uint8)
                values[y,x] = skill.get_action_value(state)[1][0]
            self.skill_title = skill_title
            self.skill_image = values
        return self.render_env(values=values, render_mode=self.render_mode, task_title=self.render_params["task_title"], skill_title=self.render_params["skill_title"], dpi=self.render_params["dpi"])
    
    def _draw_action(self, ax, x, y, action, color='black'):
        if action == self.actions["up"]:    x += 0.5; y += 1; dx = 0; dy = -0.4
        if action == self.actions["right"]: y += 0.5; dx = 0.4; dy = 0
        if action == self.actions["down"]:  x += 0.5; dx = 0; dy = 0.4
        if action == self.actions["left"]:  x += 1; y += 0.5; dx = -0.4; dy = 0
        ax.add_patch(ax.arrow(x, y, dx, dy, fc=color, ec=color, width=0.005, head_width=0.4))

    def _map_init(self):
        self.grid = []
        lines = self.MAP.split('\n')
        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError("Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1": self.walls.append((i, j))
                else: self.possiblePositions.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

    def _gridmap_to_img(self):
        row_size, col_size = len(self.grid), len(self.grid[0])
        img = np.zeros([row_size, col_size, 3])
        for i in range(row_size):
            for j in range(col_size):  
                img[i:(i + 1), j:(j + 1)] = COLOURS[self.grid[i][j]]
        return img
