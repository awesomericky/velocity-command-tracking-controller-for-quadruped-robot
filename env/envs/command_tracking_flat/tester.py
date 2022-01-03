from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import command_tracking_flat
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_tracking_result
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import numpy as np
import datetime
import pdb
import random


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# user command samping
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

# # create environment from the configuration file
# cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

env = VecEnv(command_tracking_flat.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs  # include command dimension
act_dim = env.num_acts

command_period_steps = math.floor(cfg['environment']['command_period'] / cfg['environment']['control_dt'])

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    loaded_graph.to(device)

    env.load_scaling(weight_dir, int(iteration_number))
    env.initialize_n_step()
    env.reset()
    env.turn_on_visualization()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_test"+'.mp4')

    # max_steps = 1000000
    max_steps = 3000 ## 30 secs
    command_trajectory = []
    real_trajectory = []

    pdb.set_trace()

    for step in range(max_steps):
        frame_start = time.time()
        if step % command_period_steps == 0:
            sample_user_command = user_command.uniform_sample_evaluate()
            env.set_user_command(sample_user_command)

        obs, non_obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).to(device))
        _, dones = env.step(action_ll.cpu().detach().numpy())
        frame_end = time.time()

        if dones.all():
            env.reset()

        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        # command tracking logging
        command_trajectory.append(sample_user_command[0])
        real_trajectory.append([non_obs[0, 18], non_obs[0, 19], non_obs[0, 23]])

        if wait_time > 0.:
            time.sleep(wait_time)

    env.turn_off_visualization()
    env.stop_video_recording()

    command_trajectory = np.array(command_trajectory)
    real_trajectory = np.array(real_trajectory)

    plot_command_tracking_result(command_trajectory, real_trajectory, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test1010', control_dt=cfg['environment']['control_dt'])

    print("Finished at the maximum visualization steps")
