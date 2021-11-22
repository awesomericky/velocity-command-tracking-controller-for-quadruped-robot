from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_free
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_tracking_result, plot_contact_result, plot_torque_result, plot_joint_velocity_result
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import numpy as np
import datetime
import pdb


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# user command samping
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

env = VecEnv(lidar_model_free.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
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
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_test"+'.mp4')

    # max_steps = 1000000
    max_steps = 3000 ## 30 secs

    # command tracking logging initialize
    command_trajectory = []
    real_trajectory = []

    # contact logging initialize
    contact_log = np.zeros((4, max_steps), dtype=np.float32) # 0: FR, 1: FL, 2: RR, 3:RL
    torque_log = np.zeros((12, max_steps), dtype=np.float32)
    joint_velocity_log = np.zeros((12, max_steps), dtype=np.float32)

    for step in range(max_steps):
        if step % command_period_steps == 0:
            sample_user_command = user_command.uniform_sample_evaluate()
            # sample_user_command[:, 2] = 0  # set yaw rate command to zero
            env.set_user_command(sample_user_command)

        time.sleep(0.01)
        obs, non_obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]

        # command tracking logging
        command_trajectory.append(sample_user_command[0])
        real_trajectory.append([non_obs[0, 18], non_obs[0, 19], non_obs[0, 23]])

        # contact logging
        env.contact_logging()
        contact_log[:, step] = env.contact_log[0, :]

        # torque logging
        env.torque_and_velocity_logging()
        torque_log[:, step] = env.torque_and_velocity_log[0, :12]
        joint_velocity_log[:, step] = env.torque_and_velocity_log[0, 12:]

        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    command_trajectory = np.array(command_trajectory)
    real_trajectory = np.array(real_trajectory)
    plot_command_tracking_result(command_trajectory, real_trajectory, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test', control_dt=cfg['environment']['control_dt'])
    plot_contact_result(contact_log, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test', control_dt=cfg['environment']['control_dt'])
    plot_torque_result(torque_log, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test', control_dt=cfg['environment']['control_dt'])
    plot_joint_velocity_result(joint_velocity_log, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test', control_dt=cfg['environment']['control_dt'])

    env.turn_off_visualization()
    env.stop_video_recording()

    print("Finished at the maximum visualization steps")