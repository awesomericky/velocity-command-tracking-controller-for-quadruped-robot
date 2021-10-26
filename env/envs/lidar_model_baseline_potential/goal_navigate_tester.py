import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_baseline_potential
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_enviroment_model_param, UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_result
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
import datetime
from collections import Counter
import argparse
from collections import defaultdict
import pdb
from raisimGymTorch.env.envs.lidar_model.model import Lidar_environment_model
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_normal, Stochastic_action_planner_uniform_bin
from raisimGymTorch.env.envs.lidar_model.action import Zeroth_action_planner, Modified_zeroth_action_planner, Stochastic_action_planner_uniform_bin_baseline
from raisimGymTorch.env.envs.lidar_model.storage import Buffer


def transform_coordinate_LW(w_init_coordinate, l_coordinate_traj):
    """
    Transform LOCAL frame coordinate trajectory to WORLD frame coordinate trajectory
    (LOCAL frame --> WORLD frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param l_coordinate_traj: coordintate trajectory in LOCAL frame (n_step, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    w_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix)
    w_coordinate_traj += w_init_coordinate[:, :-1]
    return w_coordinate_traj

def transform_coordinate_WL(w_init_coordinate, w_coordinate_traj):
    """
    Transform WORLD frame coordinate trajectory to LOCAL frame coordinate trajectory
    (WORLD frame --> LOCAL frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param w_coordinate_traj: coordintate trajectory in WORLD frame (n_step, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    l_coordinate_traj = w_coordinate_traj - w_init_coordinate[:, :-1]
    l_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix.T)
    return l_coordinate_traj

np.random.seed(8)

# task specification
task_name = "lidar_environment_model"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-tw', '--tracking_weight', help='pre-trained command tracking policy weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
command_tracking_weight_path = args.tracking_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# user command sampling
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

# create environment from the configuration file
env = VecEnv(lidar_model_baseline_potential.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Evaluating
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['data_collection']['command_period'] / cfg['environment']['control_dt'])
evaluate_command_sampling_steps = math.floor(cfg['evaluating']['command_period'] / cfg['environment']['control_dt'])

state_dim = cfg["architecture"]["state_encoder"]["input"]
command_dim = cfg["architecture"]["command_encoder"]["input"]
P_col_dim = cfg["architecture"]["traj_predictor"]["collision"]["output"]
coordinate_dim = cfg["architecture"]["traj_predictor"]["coordinate"]["output"]   # Just predict x, y coordinate (not yaw)

command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))


start = time.time()

env.initialize_n_step()
env.reset()
goal_position = env.set_goal()[np.newaxis, :]
env.turn_on_visualization()
# env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + "lidar_2d_normal_sampling" + '.mp4')

# command tracking logging initialize
command_traj = []

# Initialize number of steps
step = 0
n_test_case = 0
n_success_test_case = 0
num_goals = 25

# MUST safe period from collision
MUST_safety_period = 2.
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
sample_user_command = np.zeros(3)
prev_coordinate_obs = np.zeros((1, 3))
goal_rewards = np.zeros((cfg["evaluating"]["number_of_sample"], 1), dtype=np.float32)

# pdb.set_trace()

while n_test_case < num_goals:
    frame_start = time.time()
    new_action_time = step % command_period_steps == 0

    obs, _ = env.observe(False)  # observation before taking step

    init_coordinate_obs = env.coordinate_observe()

    if new_action_time:
        # compute reward (goal reward + safety reward)
        goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))

        v_magnitude = 1.0
        computed_heading_direction = env.observe_potential_heading_direction()
        computed_yaw = np.arctan2(computed_heading_direction[1], computed_heading_direction[0])
        heading_direction = computed_heading_direction / ((computed_heading_direction[0]**2 + computed_heading_direction[1]**2)**.5)

        # if (- np.pi * 2/5 <= computed_yaw) and (computed_yaw <= np.pi * 2/5):
        #     sample_user_command[2] = np.clip(1.0 * computed_yaw, -1.0, 1.0)
        #     sample_user_command[0] = np.clip((v_magnitude**2 - sample_user_command[2]**2)**0.5, -0.5, 0.5)
        #     sample_user_command[1] = 0
        # elif (- np.pi * (2/5 + 1/5) >= computed_yaw) or (computed_yaw >= np.pi * (2/5 + 1/5)):
        #     sample_user_command[2] = np.clip(- np.sign(computed_yaw) * 1.0 * (np.pi - abs(computed_yaw)), -1.0, 1.0)
        #     sample_user_command[0] = np.clip(-(v_magnitude**2 - sample_user_command[2]**2)**0.5, -0.5, 0.5)
        #     sample_user_command[1] = 0
        # else:
        #     v_magnitude = 0.4
        #     sample_user_command[0] = heading_direction[0] * v_magnitude
        #     sample_user_command[1] = heading_direction[1] * v_magnitude
        #     sample_user_command[2] = 0

        # if (- np.pi * 1/3 <= computed_yaw) and (computed_yaw <= np.pi * 1/3):
        #     sample_user_command[2] = np.clip(1.0 * computed_yaw, -1.0, 1.0)
        #     sample_user_command[0] = np.clip((v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0
        # elif (- np.pi * 2/3 >= computed_yaw) or (computed_yaw >= np.pi * 2/3):
        #     sample_user_command[2] = np.clip(- np.sign(computed_yaw) * 1.0 * (np.pi - abs(computed_yaw)), -1.0, 1.0)
        #     sample_user_command[0] = np.clip(-(v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0
        # else:
        #     sample_user_command[2] = np.clip(1.0 * computed_yaw, -1.0, 1.0)
        #     sample_user_command[0] = 0
        #     sample_user_command[1] = 0

        if (- np.pi * 2/5 <= computed_yaw) and (computed_yaw <= np.pi * 2/5):
            sample_user_command[2] = np.clip(computed_yaw / (cfg['data_collection']['command_period'] / 1), -1.0, 1.0)
            sample_user_command[0] = np.clip((v_magnitude**2 - sample_user_command[2]**2)**0.5, -0.7, 0.7)
            # sample_user_command[0] = 0.7
            sample_user_command[1] = 0
        elif (- np.pi * 3/5 >= computed_yaw) or (computed_yaw >= np.pi * 3/5):
            sample_user_command[2] = np.clip(- np.sign(computed_yaw) * (np.pi - abs(computed_yaw)) / (cfg['data_collection']['command_period'] / 1), -1.0, 1.0)
            sample_user_command[0] = np.clip(-(v_magnitude**2 - sample_user_command[2]**2)**0.5, -0.7, 0.7)
            # sample_user_command[0] = - 0.7
            sample_user_command[1] = 0
        else:
            sample_user_command[2] = np.clip(computed_yaw / (cfg['data_collection']['command_period'] / 1), -1.0, 1.0)
            sample_user_command[0] = 0
            sample_user_command[1] = 0

        # if (- np.pi * 1/3 <= computed_yaw) and (computed_yaw <= np.pi * 1/3):
        #     sample_user_command[2] = np.clip(0.76 * computed_yaw, -1.0, 1.0)
        #     sample_user_command[0] = np.clip((v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0
        # elif (- np.pi * 2/3 >= computed_yaw) or (computed_yaw >= np.pi * 2/3):
        #     sample_user_command[2] = np.clip(- np.sign(computed_yaw) * 0.76 * (np.pi - abs(computed_yaw)), -1.0, 1.0)
        #     sample_user_command[0] = np.clip(-(v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0
        # else:
        #     v_magnitude = 0.4
        #     sample_user_command[0] = heading_direction[0] * v_magnitude
        #     sample_user_command[1] = heading_direction[1] * v_magnitude
        #     sample_user_command[2] = 0

        # if (- np.pi / 2 <= computed_yaw) and (computed_yaw <= np.pi / 2):
        #     sample_user_command[2] = np.clip(0.76 * computed_yaw, -1.0, 1.0)
        #     sample_user_command[0] = np.clip((v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0
        # elif (- np.pi / 2 > computed_yaw) or (computed_yaw > np.pi / 2):
        #     sample_user_command[2] = np.clip(- np.sign(computed_yaw) * 0.76 * (np.pi - abs(computed_yaw)), -1.0, 1.0)
        #     sample_user_command[0] = np.clip(-(v_magnitude**2 - sample_user_command[2]**2)**0.5, -1.0, 1.0)
        #     sample_user_command[1] = 0

    tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
    tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
    tracking_obs = tracking_obs.astype(np.float32)

    with torch.no_grad():
        tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))

    _, done = env.step(tracking_action.cpu().detach().numpy())

    step += 1

    # Command logging
    command_traj.append(sample_user_command)

    frame_end = time.time()

    # # # (2000 sample, 10 bin ==> 0.008 sec)
    # print(frame_end - frame_start)
    # if new_action_time:
    #     print(f"Action: {frame_end - frame_start}")

    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

    if wait_time > 0.:
        time.sleep(wait_time)

    # fail
    if done[0] == True:
        env.initialize_n_step()
        env.reset()
        goal_position = env.set_goal()[np.newaxis, :]
        n_test_case += 1
        step = 0
        command_traj = []
        sample_user_command = np.zeros(3)
    # success
    elif current_goal_distance < 0.5:
        # plot command trajectory
        command_traj = np.array(command_traj)

        # reset action planner and set new goal
        goal_position = env.set_goal()[np.newaxis, :]
        n_test_case += 1
        step = 0
        command_traj = []
        sample_user_command = np.zeros(3)
        n_success_test_case += 1

print(f"Result : {n_success_test_case} / {num_goals}")

# env.stop_video_recording()
env.turn_off_visualization()

