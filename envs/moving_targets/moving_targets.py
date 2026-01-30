import collections, os, random, textwrap, copy
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from pygame.compat import geterror
from pygame.locals import QUIT
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})

import cv2
cv2.ocl.setUseOpenCL(False)

main_dir = os.path.split(os.path.abspath(__file__))[0]
assets_dir = os.path.join(main_dir, 'assets')
# os.environ["SDL_VIDEODRIVER"] = "dummy"

def _load_image(name):
    fullname = os.path.join(assets_dir, name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert_alpha()
    return image


def _calculate_topleft_position(position, sprite_size):
    return sprite_size * position[1], sprite_size * position[0]


class _Collectible(pygame.sprite.Sprite):
    _COLLECTIBLE_IMAGES = {
        ('square', 'purple'): 'purple_square.png',
        ('circle', 'purple'): 'purple_circle.png',
        ('square', 'beige'): 'beige_square.png',
        ('circle', 'beige'): 'beige_circle.png',
        ('square', 'blue'): 'blue_square.png',
        ('circle', 'blue'): 'blue_circle.png'
    }

    def __init__(self, sprite_size, shape, colour):
        self.name = shape + '_' + colour
        self._sprite_size = sprite_size
        self.shape = shape
        self.colour = colour
        pygame.sprite.Sprite.__init__(self)
        image = _load_image(self._COLLECTIBLE_IMAGES[(self.shape, self.colour)])
        self.high_res = image
        self.low_res = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.image = self.low_res
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)


class _Player(pygame.sprite.Sprite):
    def __init__(self, sprite_size):
        self.name = 'player'
        self._sprite_size = sprite_size
        pygame.sprite.Sprite.__init__(self)
        image = _load_image('character.png')
        self.high_res = image
        self.low_res = pygame.transform.scale(image, (sprite_size, sprite_size))
        self.image = self.low_res
        self.rect = self.image.get_rect()
        self.position = None

    def reset(self, position):
        self.position = position
        self.rect.topleft = _calculate_topleft_position(position, self._sprite_size)

    def step(self, move):
        self.position = (self.position[0] + move[0], self.position[1] + move[1])
        self.rect.topleft = _calculate_topleft_position(self.position, self._sprite_size)
    

class CollectEnv(gym.Env):
    """
    This environment consists of an agent attempting to collect a number of objects. The agents has four actions
    to move him up, down, left and right, but may be impeded by walls.
    There are two types of objects in the environment: fridges and TVs, each of which take one of three colours
    (white, blue and purple) for a total of six objects.

    The objects the agent must collect can be specified by passing a goal condition lambda to the environment.

    """

    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    _BOARDS = {
        'original': ['##########',
                     '#        #',
                     '#        #',
                     '#    #   #',
                     '#   ##   #',
                     '#  ##    #',
                     '#   #    #',
                     '#        #',
                     '#        #',
                     '##########'],
    }

    _AVAILABLE_COLLECTIBLES = [
        ('square', 'purple'),
        ('circle', 'purple'),
        ('square', 'beige'),
        ('circle', 'beige'),
        ('square', 'blue'),
        ('circle', 'blue')
    ]

    _WALL_IMAGE = 'wall.png'
    _GROUND_IMAGE = 'ground.png'

    _ACTIONS = {
        0: (-1, 0),  # North
        1: (0, 1),  # East
        2: (1, 0),  # South
        3: (0, -1),  # West
    }

    SPRITE_SIZE = 50

    def __init__(self, board='original', obs_scale=5, available_collectibles=None, start_positions=None, random_objects=False, random_player=True, seed=None, render_mode="rgb_array"):
        """
        Create a new instance of the RepoMan environment. The observation space is a single RGB image of size 400x400,
        and the action space are four discrete actions corresponding to North, East, South and West.


        :param board: the state of the walls and free space
        :param available_collectibles: the items that will be placed in the task
        :param start_positions: an option parameter to specify the starting position of the objects and player
        By default, the episode ends when any object is collected.
        """

        if render_mode!="human": os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.steps = 0
        self.viewer = None
        self.close = False
        self.render_mode = render_mode
        self.start_positions = start_positions
        pygame.init()
        pygame.display.init()
        pygame.display.set_mode((1, 1))

        self.object_types = ["beige","blue","purple","square","circle"]
        self.predicates = sorted(self.object_types)
        # self.constraints = []

        self.obs_scale = obs_scale
        self.board = np.array([list(row) for row in self._BOARDS[board]])
        self.free_spaces = list(map(tuple, np.argwhere(self.board != '#')))
        self.SCREEN_SIZE = (self.board.shape[0]*self.SPRITE_SIZE, self.board.shape[1]*self.SPRITE_SIZE)
        self._bestdepth = pygame.display.mode_ok(self.SCREEN_SIZE, 0, 32)
        self.surface = pygame.Surface(self.SCREEN_SIZE, 0, self._bestdepth)
        self._background = pygame.Surface(self.SCREEN_SIZE)
        self._clock = pygame.time.Clock()
        self._build_board()

        self.initial_positions = None
        self.collectibles = pygame.sprite.Group()
        self.collected = pygame.sprite.Group()
        self.render_group = pygame.sprite.RenderPlain()
        self.player = _Player(self.SPRITE_SIZE)
        self.render_group.add(self.player)
        self.random_player = random_player
        self.random_objects = random_objects

        self.available_collectibles = available_collectibles \
            if available_collectibles is not None else self._AVAILABLE_COLLECTIBLES
        for shape, colour in self.available_collectibles:
            self.collectibles.add(_Collectible(self.SPRITE_SIZE, shape, colour))
        
        # Gym spaces for observation and action space
        self.observation_space = Box(low=0, high=255, shape=[self.SCREEN_SIZE[0]//self.obs_scale, self.SCREEN_SIZE[1]//self.obs_scale, 3], dtype=np.uint8)
        self.actions = dict(up = 0, right = 1, down = 2, left = 3)
        self.action_space = Discrete(len(self.actions),seed=seed)

    def get_predicates(self):
        predicates = np.zeros(len(self.predicates), dtype=np.uint8)
        for i,s in enumerate(self.predicates):
            if self.collected and s in [self.collected[0].shape,self.collected[0].colour]:
                    predicates[i] = 1
        return predicates

    def _build_board(self):
        for col in range(self.board.shape[1]):
            for row in range(self.board.shape[0]):
                position = _calculate_topleft_position((row, col), self.SPRITE_SIZE)
                image = self._WALL_IMAGE if self.board[row, col] == '#' else self._GROUND_IMAGE
                image = _load_image(image)
                image = pygame.transform.scale(image, (self.SPRITE_SIZE, self.SPRITE_SIZE))
                self._background.blit(image, position)

    def draw_screen(self, surface,draw_background=True, size=None):
        if draw_background: surface.blit(self._background, (0, 0))
        else:               self._background.fill((0,0,0))
        self.render_group.draw(surface)
        if size: surface = pygame.transform.scale(surface, size)
        surface_array = pygame.surfarray.array3d(surface)
        observation = np.array(surface_array, dtype=np.uint8).swapaxes(0, 1)
        del surface_array
        return observation

    def get_obs(self):
        return self.draw_screen(self.surface,size=self.observation_space.shape[:2])

    def reset(self, seed=None, **kwargs):
        self.steps = 0
        self.goal = None
        self.collected = []
        self._build_board()
        self.render_group.empty()
        self.render_group.add(self.collectibles.sprites())
        self.render_group.add(self.player)
        if not self.random_objects: seed = 0
        self.np_random, self.seed = seeding.np_random(seed)
        # self.np_random, self.seed = seeding.np_random(int(self.np_random.choice(range(0,1))))
        
        for sprite in self.collectibles:            
            sprite.image = sprite.low_res
            self.player.image = self.player.low_res

        self.free_spaces = set(map(tuple, np.argwhere(self.board != '#')))
        # self.free_spaces = set(map(tuple, self.np_random.choice(self.free_spaces, size=len(self.render_group)+1, replace=False)))
        if self.start_positions is None:
            positions = list(map(tuple, self.np_random.choice(list(self.free_spaces), size=len(self.render_group), replace=False)))
        else:
            start_positions = collections.OrderedDict(sorted(self.start_positions.items()))
            positions = start_positions.values()
        self.free_spaces = self.free_spaces - set(positions)

        self.initial_positions = collections.OrderedDict()               
        render_group = sorted(self.render_group, key=lambda x: x.name) 
        for position, sprite in zip(positions, render_group):
            self.initial_positions[sprite] = position
            sprite.reset(position)
        if not self.random_objects and self.random_player:
            position = random.choice(list(self.free_spaces))
            self.free_spaces = self.free_spaces - set([position])
            self.free_spaces.add(self.player.position)
            self.initial_positions[self.player] = position
            self.player.reset(position)
                
        obs = self.draw_screen(self.surface,size=self.observation_space.shape[:2])
        return obs, {}

    def step(self, action):    
        self.collected = []
        direction = self._ACTIONS[action]
        prev_pos = self.player.position
        next_pos = (direction[0] + prev_pos[0], direction[1] + prev_pos[1])
        if self.board[next_pos] != '#':
            self.player.step(direction)
        
        self.collected = pygame.sprite.spritecollide(self.player, self.collectibles, False)
        if self.collected and self.random_objects:
            position = tuple(self.np_random.choice(list(self.free_spaces)))
            self.free_spaces = self.free_spaces - set([position])
            self.free_spaces.add(self.player.position)
            self.collected[0].reset(position)
             
        obs = self.get_obs()
        return obs, 0, False, False, {}

    def render(self, *args, **kwargs):
        if self.close:
            if self.viewer is not None:
                pygame.quit()
                self.viewer = None
            return
        
        if self.viewer is None:
            self.viewer = pygame.display.set_mode(self.SCREEN_SIZE, 0, self._bestdepth)

        self._clock.tick(10 if self.render_mode != 'human' else 2)
        arr = self.draw_screen(self.viewer)
        pygame.display.flip()
        return arr


class MovingTargets(gym.core.ObservationWrapper):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}
    def __init__(self, random_objects=True, **kwargs):
        """The moving targets domain"""
        super().__init__(CollectEnv(random_objects=random_objects, **kwargs))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=[84, 84, 3], dtype=np.uint8)

        self.render_params = dict(dpi=200, skill=None, state=None, title=None, task_title=None, skill_title=None)
        self.skill_image = np.zeros(self.board.shape), np.zeros(self.board.shape)+float("-inf")
        self.skill_title = ""; self.fig=None

    def observation(self, obs): 
        """Make the observations egocentric""" 
         
        s = obs.shape[0]
        st = self.env.unwrapped.SPRITE_SIZE//self.obs_scale
        y, x = self.env.unwrapped.player.position[0]*st, self.env.unwrapped.player.position[1]*st

        # Egocentric position
        ox = s//2-x    
        rgb_img = obs.copy()
        if ox>=0:
            rgb_img[:,ox:s//2,:] = obs[:,:x,:]    
            rgb_img[:,s//2:,:] = obs[:,x:x+s//2+s%2,:]   
            rgb_img[:,:ox,:] = obs[:,x+s//2+s%2:,:]   
        else:
            ox = s+ox
            rgb_img[:,s//2:ox,:] = obs[:,x:,:]    
            rgb_img[:,:s//2,:] = obs[:,x-s//2:x,:]   
            rgb_img[:,ox:,:] = obs[:,:x-s//2,:]    
        obs = rgb_img.copy()
        rgb_img[s-(y+st):,:,:] = obs[:y+st,:,:] 
        rgb_img[:s-(y+st),:,:] = obs[y+st:,:,:] 
        
        return cv2.resize(rgb_img, dsize=self.observation_space.shape[:2], interpolation=cv2.INTER_AREA)
    
    def render(self, *args, **kwargs):    
        if self.render_params["skill"]: 
            return self.render_skill(self.render_params["skill"], self.render_params["state"], self.render_params["task_title"], self.render_params["skill_title"], self.render_params["dpi"])
        else:                           
            return self.render_env(*args, **kwargs)

    def render_env(self, *args, **kwargs):   
        return self.env.render(*args, **kwargs)

    def render_skill(self, skill, state, task_title="", skill_title="", dpi=100): 
        if not self.fig: 
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi, facecolor='w', edgecolor='k')        
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.fig = fig, (ax1, ax2), divider, cax       
        fig, (ax1, ax2), divider, cax = self.fig
        env_render = self.render_env()
        if self.render_mode=="rgb_array": ax1.imshow(env_render)
        if task_title: ax1.set_title(r" \\ ".join(textwrap.wrap(task_title, width=80)))
        ax1.axis('off')
        ax1.set_aspect('equal', 'box')
            
        state = copy.deepcopy(state)
        position = self.player.position
        actions, values = self.skill_image
        if True: # self.skill_title != skill_title:
            actions, values = np.zeros(self.board.shape), np.zeros(self.board.shape)+float("-inf")  
            for x in range(self.env.board.shape[0]):
                for y in range(self.env.board.shape[1]):
                    # if (y,x) not in self.free_spaces: continue
                    self.env.player.reset((y,x))
                    state["env_state"][0] = self.observation(self.env.get_obs())
                    action, value = skill.get_action_value(state)
                    actions[y,x], values[y,x] = action[0], value[0]  
                    # self._draw_action(ax2, x, y, action[0])
            self.player.reset(position)
            self.skill_title = skill_title
            self.skill_image = actions, values

        # ax2.imshow(self.map_img, origin="upper", extent=[0, self.n, self.m, 0])
        c = ax2.imshow(values, origin="upper", cmap="YlOrRd", extent=[0, self.board.shape[0], self.board.shape[1], 0]) 
        plt.colorbar(c, cax=cax)
        ax2.axis('off')
        ax2.set_aspect('equal', 'box')
        if skill_title: ax2.set_title(r" \\ ".join(textwrap.wrap(skill_title, width=80)))
        plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.05, wspace=0, hspace=0) 
              
        # plt.tight_layout()      
        if self.render_mode == 'rgb_array':
            fig.canvas.draw()
            height, width = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            plt.close(fig); self.fig = None
            return img
        else:
            plt.pause(0.001) 
    
    def _draw_action(self, ax, x, y, action, color='black'):
        if action == self.actions["up"]:    x += 0.5; y += 1; dx = 0; dy = -0.4
        if action == self.actions["right"]: y += 0.5; dx = 0.4; dy = 0
        if action == self.actions["down"]:  x += 0.5; dx = 0; dy = 0.4
        if action == self.actions["left"]:  x += 1; y += 0.5; dx = -0.4; dy = 0
        ax.add_patch(ax.arrow(x, y, dx, dy, fc=color, ec=color, width=0.005, head_width=0.4))