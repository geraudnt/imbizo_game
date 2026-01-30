import numpy as np, gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from safety_gym.envs.engine import Engine
import textwrap

"""
Installing libraries:
- conda create --name sm python==3.8
- conda activate sm
- pip install -r requirements.txt
- conda install -c conda-forge spot

Installing mujoco:
- Needed libraries and Environment variables
- conda install patchelf
- conda install -c menpo osmesa
- wget https://rpmfind.net/linux/atrpms/sl6-x86_64/atrpms/testing/libgcrypt11-1.4.0-15.el6.x86_64.rpm
- conda install conda-forge::rpm-tools
- rpm2cpio libgcrypt11-1.4.0-15.el6.x86_64.rpm | cpio -id  # This will create a usr folder in the current directory.
- mv usr/lib64/* CONDA_ENV_PATH/lib/   # e.g. mv usr/lib64/* ~/miniconda3/envs/sm/lib
- export CPATH=CONDA_ENV_PATH/include
- export C_INCLUDE_PATH=:CONDA_ENV_PATH/include 
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
- export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:CONDA_ENV_PATH/lib
- export LDFLAGS="-L/CONDA_ENV_PATH/lib"
- You can also add the above to ~/.bashrc

- Installing mujoco by largely folowing the instructions from https://github.com/openai/mujoco-py
- Download the MuJoCo version 2.1 binaries for Linux (https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz).
- Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210

Installing Safety_gym:
- cd envs/safety_gym/safety-gym
- pip install -e .
"""


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})

def rot2quat(theta):
    ''' Get a quaternion rotated only about the Z axis '''
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')

class SafetyEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array'], "render_fps": 10}

    d = 1.4

    config = {
        'num_steps': 1000000,
        'robot_base': 'xmls/point.xml',
        'task': 'button',
        'render_lidar_markers': True,
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'observe_buttons': True,
        'constrain_hazards': False,
        'constrain_buttons': False,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 2,
        'buttons_num': 4,
        'continue_goal': True,    
    }

    def __init__(self, config={}, use_dense=False, render_mode="human"):
        
        self.sensors = sorted(["hazards_lidar","goal_lidar","buttons_lidar"])
        self.predicates = sorted(["hazards","goal","buttons"])
        self.constraints = ["hazards"]
        self.threshold = 0.9
        self.use_dense = use_dense
        self.render_mode = render_mode
        self.render_params = dict(dpi=200, skill=None, state=None, title=None, task_title=None, skill_title=None)

        self.config.update(config)
        self.env = Engine(self.config)
        self.all_states, self.skill_title, self.skill_image = None, None, None
        self.agent_pos, self.agent_rot = (0,0), 0

        self.observation_space = gym.spaces.Box(low=self.env.observation_space.low, high=self.env.observation_space.high, shape=(self.env.observation_space.shape[0],), dtype=self.env.observation_space.dtype)
        self.action_space = gym.spaces.Box(low=self.env.action_space.low, high=self.env.action_space.high, shape=(self.env.action_space.shape[0],), dtype=self.env.action_space.dtype)
        self.actions = dict(up = np.array([0.1,0]), down = np.array([-0.1,0]), right = np.array([0,-1]), left = np.array([0,1]))

    def get_predicates(self):
        predicates = np.zeros(len(self.predicates), dtype=np.uint8)
        for i,s in enumerate(self.predicates):
            if np.max(self.env.curr_obs[s+"_lidar"])>=self.threshold: predicates[i] = 1
        return predicates
    
    def reset(self, **kwargs):
        self.all_states, self.skill_title, self.skill_image = None, None, None
        return self.env.reset(), {}
    
    def dist_xy(self, pos1, pos2):
        ''' Return the distance from the robot to an XY position '''
        return np.sqrt(np.sum(np.square(pos1 - pos2)))
    
    def get_reward(self, info={}):
        reward = 0
        if self.use_dense:
            rewards = []
            for i,s in enumerate(self.predicates):
                if "goal" in s:
                    dist_goal = self.dist_xy(self.env.goal_pos[:2], self.agent_pos)
                    rewards.append(np.exp(-0.5*dist_goal)-1)
                    # reward += (self.pre_dist_goal - dist_goal)# /len(self.sensors)
                    self.pre_dist_goal = dist_goal
                if "hazards" in s:
                    dist_hazards = np.min([self.dist_xy(pos[:2], self.agent_pos) for pos in self.env.hazards_pos])
                    rewards.append(np.exp(-0.5*dist_hazards)-1) 
                    # reward += (self.pre_dist_hazards - dist_hazards)# /len(self.sensors)
                    self.pre_dist_hazards = dist_hazards
                if "buttons" in s:
                    dist_buttons = np.min([self.dist_xy(pos[:2], self.agent_pos) for pos in self.env.buttons_pos])
                    rewards.append(np.exp(-0.5*dist_buttons)-1) 
                    # reward += (self.pre_dist_buttons - dist_buttons)# /len(self.sensors)
                    self.pre_dist_buttons = dist_buttons
                info["reward_"+self.predicates[i]] = reward
            reward = np.max(rewards)
        return reward
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.agent_pos = self.env.world.robot_pos()[:2]
        self.agent_rot = np.arctan2(action[0], action[1])*180/np.pi
        reward = self.get_reward(info)
        return obs, reward, done, False, info
    
    def render(self, *args, **kwargs):    
        if self.render_params["skill"]: return self.render_skill(self.render_params["skill"], self.render_params["state"], task_title=self.render_params["task_title"], skill_title=self.render_params["skill_title"], dpi=self.render_params["dpi"])
        else:                           return self.render_env(*args, **kwargs)

    def render_env(self, *args, **kwargs):   
        if self.render_mode == 'rgb_array':
            return self.env.render(mode='rgb_array', camera_id=1, width=1000,height=1000)
        else:
            self.env.render()
            if self.env.viewer.cam.elevation != -60: 
                self.env.viewer.cam.distance = 10
                self.env.viewer.cam.elevation = -60
                self.env.render(); self.env.render()
    
    def render_rewards(self):
        size, D = 100, self.d*2
        rewards = np.zeros((size,size))
        gridx = np.linspace(-D,D,size)
        gridy = np.linspace(D,-D,size)
        for x in reversed(range(size)):
            for y in range(size):
                self.agent_pos = np.array((gridx[x],gridy[y]))
                rewards[y,x] = self.get_reward({})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        env_render = self.env.render(mode='rgb_array', camera_id=1, width=1000,height=1000)
        ax1.imshow(env_render)
        ax1.set_title("Environment")
        ax1.axis('off')
        ax1.set_aspect('equal', 'box')
        im = ax2.imshow(rewards, cmap='RdYlBu_r')
        ax2.set_aspect('equal', 'box')
        ax2.set_title("Rewards")
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        # fig.colorbar(cax, ax=ax2)

        tick_positions = np.arange(0, size, size//10)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{gridx[i]:.1f}' for i in tick_positions])
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels([f'{gridy[i]:.1f}' for i in tick_positions])

        plt.tight_layout()
        plt.show()

    def render_skill(self, skill, state, task_title="", skill_title="", dpi=100): 
        size, D = dpi//2, self.d*2
        values = np.zeros((size,size))
        gridx = np.linspace(-D,D,size)
        gridy = np.linspace(D,-D,size)
        
        if skill_title == self.skill_title:
            values = self.skill_image
        else:
            if type(self.all_states) == type(None):
                config = self.config.copy()
                config.update({'hazards_locations': [], 'buttons_locations': [], 'robot_locations': [(0,0)], 'robot_rot': 0})
                for name in ['hazard','button']:
                    for num in range(self.config[name+"s_num"]):
                        obj = name+str(num)
                        x,y = self.env.data.get_body_xpos(obj)[:2]
                        if 'hazard' in obj: config['hazards_locations'].append((x,y))
                        if 'button' in obj: config['buttons_locations'].append((x,y))
                self.render_env = Engine(config); self.render_env.reset()
                self.render_env.goal_button = self.env.goal_button
                
                obj_mask = np.zeros((size,size))
                states = []
                for x in range(size):
                    for y in range(size):         
                        self.render_env.sim.data.qpos[:2] = np.array([gridx[x],gridy[y]])
                        self.render_env.sim.data.qvel[:2] *= 0
                        states.append(self.render_env.obs())
                        for s in self.predicates:
                            if np.max(self.render_env.curr_obs[s+"_lidar"])>=self.threshold: obj_mask[y,x] = 1; break
                self.all_states = np.array(states), obj_mask
            states = {"env_state": self.all_states[0]}
            values = skill.get_action_value(states)[1].reshape((size,size)).T  
            values = np.ma.masked_where(self.all_states[1], values)
            self.skill_image = values
            self.skill_title = skill_title

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
        env_render = self.env.render(mode='rgb_array', camera_id=1, width=1000,height=1000)
        ax1.imshow(env_render)
        if task_title: ax1.set_title(r" \\ ".join(textwrap.wrap(task_title, width=100)), pad=10)
        
        ax1.axis('off')
        ax1.set_aspect('equal', 'box')

        im = ax2.imshow(values, cmap='RdYlBu_r')
        ax2.set_aspect('equal', 'box')
        if skill_title: ax2.set_title(r" \\ ".join(textwrap.wrap(skill_title, width=100)), pad=10)
        
        for name in ['hazard','button']:
            for num in range(self.config[name+"s_num"]):
                obj = name+str(num)
                x, y = self.env.data.get_body_xpos(obj)[:2]
                x = np.abs(gridx - x).argmin()
                y = np.abs(gridy - y).argmin()
                color, radius, alpha, zorder = "white", 6, 1, 1
                if name=="hazard": color, zorder = (0,0,1,alpha), 1
                if name=="button": color, zorder = (0,1,0,alpha), 2
                if name=="button" and self.env.goal_button==num: color, zorder = (1,0,0,alpha), 3
                ax2.add_patch(patches.Circle((x,y), radius=radius, lw=2, ec='white', fc=color, transform=ax2.transData, zorder=zorder)) 
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        tick_positions = np.arange(0, size, size//10)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{gridx[i]:.1f}' for i in reversed(tick_positions)])
        ax2.set_yticks(tick_positions)
        ax2.set_yticklabels([f'{gridy[i]:.1f}' for i in tick_positions])

        plt.tight_layout()      
        # plt.show() 
        if self.render_mode == 'rgb_array':
            fig.canvas.draw()
            height, width = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(int(width), int(height), 3)
            plt.close(fig)
            return img
        else:
            plt.pause(0.001) 
