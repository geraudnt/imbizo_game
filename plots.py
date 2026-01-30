import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import seaborn as sns
import csv
import pandas as pd
import argparse
# from scipy.interpolate import spline
from scipy.signal import savgol_filter

from sm import TaskPrimitive, SkillMachine, evaluate
from sb3_utils import TD3Agent, DQNAgent
from sm_ql import QLAgent
from rm import Task

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "text.latex.preamble":r'\usepackage{pifont,marvosym,scalerel}'
})


parser = argparse.ArgumentParser()
parser.add_argument(
    '--env',
    default= "Office-Coffee-Task-v0",
    help="Environment"
)
parser.add_argument(
    '--exp',
    default="office",
    help="Experiment. E.g office, office_iclr, movingtargets_iclr",
)
parser.add_argument(
    '--metric',
    default="eval total reward",
    help="Logged metric. E.g. eval successes",
)
args = parser.parse_args()


def plot_logs(num_runs,num_steps,m,labels,dirs):
    data = []

    for j in range(len(dirs)):
        try:
            dirr = f'baselines_iclr/data/logs/{dirs[j]}/{args.env}/'
            filer = open(dirr+'0/progress.csv')
            print(dirr+'0/progress.csv')
            csvreader = csv.reader(filer)
            all_data_pnts = [row for row in csvreader]
            episodes, steps, performance = all_data_pnts[0].index("episodes"), all_data_pnts[0].index("steps"), all_data_pnts[0].index(args.metric)
            all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
            task = all_data_pnts[:,steps]
            print("data:", all_data_pnts.shape)
            a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
            b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
            c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
            all_data_pnts = np.array([a,b,c]).T
            
            for i in range(num_runs):
                try:
                    filer = open(dirr+'/'+str(i)+'/progress.csv')
                    csvreader = csv.reader(filer)
                    data_pnts = []
                    for row in csvreader: data_pnts.append(row)
                    episodes, steps, performance = data_pnts[0].index("episodes"), data_pnts[0].index("steps"), data_pnts[0].index(args.metric)
                    # data_pnts[1][-2:]=["0","0"]
                    data_pnts = np.array(data_pnts[1:]).astype(np.float32)[:num_steps,:]
                    a = np.sum(data_pnts[:,episodes].reshape(-1, m), axis=1)
                    b = np.sum(data_pnts[:,steps].reshape(-1, m), axis=1)
                    c = np.sum(data_pnts[:,performance].reshape(-1, m), axis=1)
                    data_pnts = np.array([a,b,c]).T
                    all_data_pnts = np.dstack( [all_data_pnts, data_pnts] )
                except:
                    print(labels[j], i, "skipped")
            data.append((np.mean(all_data_pnts, axis=2)[:,2], np.std(all_data_pnts, axis=2)[:,2], labels[j]))
        except:
            print(labels[j], "skipped")
    s = 20

    rc_ = {'figure.figsize':(9,7),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    lw = 2.0

    fig, ax = plt.subplots()
    for (mean, std, label) in data:
        task = np.linspace(task.min(), task.max(), len(mean))  
        ax.plot(task, mean, lw = lw, label=label)
        ax.fill_between(task, mean - std, mean + std, alpha=0.4)

    # plt.legend(loc="lower left", bbox_to_anchor=(0,0.025))
    plt.legend()
    # ax.legend_ = None
    plt.xlabel("Steps")
    if args.metric == "eval total reward": plt.ylabel("Total Reward")
    else:                             plt.ylabel(args.metric)
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.tight_layout()
    fig.savefig(f"images/iclr_{args.env}_{args.metric}.pdf", bbox_inches='tight')
    plt.show()

def plot_moving_iclr():    
    s = 20
    rc_ = {'figure.figsize':(6,5),'axes.labelsize': 40, 'font.size': 30, 
          'legend.fontsize': 30, 'axes.titlesize': 30}
    sns.set(rc=rc_, style="darkgrid",font_scale = 3)
    # rc('text', usetex=False)

    # Initialise the MDP for the given temporal logic task
    tasks = ['MovingTargets-Blue-Task-v0', 'MovingTargets-Purple-Task-v0', 'MovingTargets-Square-Task-v0', 
             'MovingTargets-Task-1-v0', 'MovingTargets-Task-2-v0', 'MovingTargets-Task-3-v0', 'MovingTargets-Task-4-v0']
    task_envs = [gym.make(env, test=True) for env in tasks]
    task_labels = ["Blue","Purple","Square", 1, 2, 3, 4]
        
    sp_dir = f'./data/MovingTargets-v0/'
    primitive_env = TaskPrimitive(task_envs[0].environment, sb3=True)
    primitive_env.goals.update(torch.load(sp_dir+"goals")) 
    SP = {primitive: DQNAgent("wvf_"+primitive, primitive_env, save_dir=sp_dir, load=True, buffer_size=1) for primitive in ['0','1']}
    SM = SkillMachine(primitive_env, SP, vectorised=True)

    # Shortest path for n desirable objects
    # Obtained from envs/moving_targets/shortest.py. -0.1 added to account for pickup action
    tasks_goals = [[2], [2], [3], [5], [2,2,2], [3], [1,2]]
    steps_mean_std=[(5.963702359346643, 2.943079914627588),  # F.shortest(2)
                    (4.662918071039668, 2.444848035735638),  # F.shortest(3)
                    (3.921158696113324, 2.0950512577860287), # F.shortest(4)
                    (3.422186605489691, 1.8375860784979885), # F.shortest(5)
                    (3.059915937000483, 1.63931166842678)    # F.shortest(6) 
                    ]
    
    gamma=1; rmin=-0.1; rmax=2
    num_runs = 100
    data_sm = np.zeros((num_runs,len(tasks))) 
    data_random = np.zeros((num_runs,len(tasks))) 
    data_optimal = np.zeros((num_runs,len(tasks))) 
    for i in range(num_runs):
        print('run: ',i)
        for j in range(len(tasks)):
            ### Learned            
            eval_total_reward, eval_successes, eval_steps = evaluate(task_envs[j], SM=SM, epsilon=0.1, gamma=gamma, episodes=1, eval_steps=50)
            data_sm[i,j] = (eval_steps-len(tasks_goals[j]))*rmin + eval_successes*len(tasks_goals[j])*rmax
            ### Random
            eval_total_reward, eval_successes, eval_steps = evaluate(task_envs[j], epsilon=1, gamma=gamma, episodes=1, eval_steps=50)
            data_random[i,j] = (eval_steps-len(tasks_goals[j]))*rmin + eval_successes*len(tasks_goals[j])*rmax
            ### Start optimal
            G, steps, t = 0, 0, 0
            while True:
                mean, std = steps_mean_std[tasks_goals[j][t]-1]
                step = max(0,np.random.normal(loc=mean, scale=std))
                steps += step
                if steps < 50: G += (step-1)*rmin + rmax
                else:          G += (50-(steps-step))*rmin; break
                t = (t+1)%len(tasks_goals[j])
            data_optimal[i,j] = G
    
    types = ["Optimal", "Composed", "Random"]
    print(data_optimal.shape)
    data = pd.DataFrame(
    [[data_optimal[i,t] for t in range(len(tasks))]+[types[0]] for i in range(len(data_sm))] +
    [[data_sm[i,t] for t in range(len(tasks))]+[types[1]] for i in range(len(data_sm))] +
    [[data_random[i,t] for t in range(len(tasks))]+[types[2]] for i in range(len(data_sm))],
      columns=task_labels+[""])
    data = pd.melt(data, "", var_name="Tasks", value_name="Average Returns")
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(x="Tasks", y="Average Returns", hue="", data=data, linewidth=3, showfliers = False)
    plt.show()
    fig.savefig("images/moving_returns.png", bbox_inches='tight')

def plot_office_iclr():
    num_runs = 60
    num_steps = 400000
    m = 4
    labels = ['SM (Ours)','QL-SM (Ours)','CRM','CRM-RS','HRM','HRM-RS','QL',"QL-RS"]
    dirs = ['sm-ql/zeroshot','sm-ql/fewshot','crm','crm-rs','hrm','hrm-rs','rm-ql','rm-ql-rs']
    plot_logs(num_runs,num_steps,m,labels,dirs)

def plot_office():
    num_runs = 10
    num_steps = 400000//10000
    m = 1
    labels = ['SM (Ours)','SM-Pretrained (Ours)','SM-QL-Pretrained (Ours)','QL (Baseline)']
    dirs = ['sm_ql','sm_ql/zeroshot','sm_ql/fewshot','ql']
    plot_logs(num_runs,num_steps,m,labels,dirs)

def plot_sb3(algo):
    num_runs = 10
    num_steps = 1000000
    m = 1
    labels = ['SM (Ours)',f'{algo} (Baseline)']
    dirs = [(f'sm_{algo}','/wvf_1'),(algo,"/skill")]
    plot_logs(num_runs,num_steps,m,labels,dirs)

if   args.exp=="office":      plot_office()
elif args.exp=="office_iclr": plot_office_iclr()
elif args.exp=="moving_iclr": plot_moving_iclr()
else:                         plot_sb3(args.exp)
