import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_test_baseline
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
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_normal, Stochastic_action_planner_uniform_bin, Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal
from raisimGymTorch.env.envs.lidar_model.action import Zeroth_action_planner, Modified_zeroth_action_planner, Stochastic_action_planner_uniform_bin_baseline
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
import random


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

# set seed
evaluate_seed = 37 # 37, 143, 534, 792, 921
random.seed(evaluate_seed)
np.random.seed(evaluate_seed)
torch.manual_seed(evaluate_seed)

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
env = VecEnv(lidar_model_test_baseline.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)
# pdb.set_trace()

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

# Load action planner
n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
action_planner = Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal(command_range=cfg["environment"]["command"],
                                                                                 n_sample=cfg["evaluating"]["number_of_sample"],
                                                                                 n_horizon=n_prediction_step,
                                                                                 n_bin=cfg["evaluating"]["number_of_bin"],
                                                                                 beta=cfg["evaluating"]["beta"],
                                                                                 gamma=cfg["evaluating"]["gamma"],
                                                                                 sigma=cfg["evaluating"]["sigma"],
                                                                                 noise_sigma=0.1,
                                                                                 noise=False,
                                                                                 action_dim=user_command_dim,
                                                                                 random_command_sampler=user_command)

env.initialize_n_step()
env.reset()
action_planner.reset()
goal_position = env.set_goal()[np.newaxis, :]
env.turn_on_visualization()
# env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + "lidar_2d_normal_sampling" + '.mp4')

# command tracking logging initialize
command_traj = []

# Initialize number of steps
step = 0
n_test_case = 0
if cfg["environment"]["type"] == 2:
    num_goals = 3
elif cfg["environment"]["type"] == 10:
    num_goals = 4
else:
    num_goals = 1

# MUST safe period from collision
MUST_safety_period = 3.0
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
sample_user_command = np.zeros(3)
goal_rewards = np.zeros((cfg["evaluating"]["number_of_sample"], 1), dtype=np.float32)
collision_idx_list = np.zeros((cfg["evaluating"]["number_of_sample"], 1), dtype=np.float32)

goal_distance_threshold = 10

# Needed for computing real time factor
total_time = 0
total_n_step = 0
num_collision_idx = []

pdb.set_trace()

while n_test_case < num_goals:
    frame_start = time.time()
    new_action_time = step % command_period_steps == 0

    # observation before taking step
    obs, _ = env.observe(False)

    if new_action_time:
        # sample command sequences
        action_candidates = action_planner.sample()
        action_candidates = np.reshape(action_candidates, (action_candidates.shape[0], -1))
        action_candidates = action_candidates.astype(np.float32)

        # prepare state
        init_coordinate_obs = env.coordinate_observe()

        # compute reward (goal reward)
        goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
        goal_position_L *= np.clip(goal_distance_threshold / current_goal_distance, a_min=None, a_max=1.)
        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))

        goal_rewards, collision_idx_list = env.baseline_compute_reward(action_candidates, np.swapaxes(goal_position_L, 0, 1), goal_rewards, collision_idx_list,
                                                                       n_prediction_step, cfg["data_collection"]["command_period"], MUST_safety_period)
        coll_idx = np.where(collision_idx_list == 1)[0]
        goal_rewards -= np.min(goal_rewards)
        goal_rewards /= (np.max(goal_rewards) + 1e-5)

        reward = 1.0 * np.squeeze(goal_rewards, -1)

        # exclude trajectory that collides with obstacle
        if len(coll_idx) != cfg["evaluating"]["number_of_sample"]:
            reward[coll_idx] = 0  # exclude trajectory that collides with obstacle

        # optimize command sequence
        cand_sample_user_command, sample_user_command_traj = action_planner.action(reward)
        sample_user_command = cand_sample_user_command.copy()

    tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
    tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
    tracking_obs = tracking_obs.astype(np.float32)

    with torch.no_grad():
        tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))

    _, done = env.step(tracking_action.cpu().detach().numpy())

    step += 1

    # reset COM buffer for terminated environment
    if done[0] == True:
        num_collision_idx.append(step)
        # print("Failed")
        # break
        # env.reset()
        # action_planner.reset()
        # sample_user_command = np.zeros(3)
        # step = 0

    # Command logging
    command_traj.append(sample_user_command)

    frame_end = time.time()

    # # (2000 sample, 10 bin ==> 0.008 sec)
    # print(frame_end - frame_start)
    # if new_action_time:
    #     print(f"Action: {frame_end - frame_start}")

    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

    if cfg["realistic"]:
        if wait_time > 0.:
            time.sleep(wait_time)

    if wait_time > 0:
        total_time += cfg['environment']['control_dt']
    else:
        total_time += (frame_end - frame_start)
    total_n_step += 1

    if current_goal_distance < 0.5:
        if cfg["environment"]["visualize_path"] and n_test_case == (num_goals - 1):
            pdb.set_trace()
        # plot command trajectory
        command_traj = np.array(command_traj)

        # reset action planner and set new goal
        action_planner.reset()
        goal_position = env.set_goal()[np.newaxis, :]
        n_test_case += 1
        step = 0
        command_traj = []
        sample_user_command = np.zeros(3)

print(f"Time: {total_time}")
print(f"Total number of steps: {total_n_step}")

num_collision = 0
for i in range(len(num_collision_idx) - 1):
    if num_collision_idx[i+1] - num_collision_idx[i] != 1:
        num_collision += 1

if len(num_collision_idx) == 0:
    num_collision = 0
else:
    num_collision += 1

print(f"Collision: {num_collision}")
print(num_collision_idx)

# env.stop_video_recording()
env.turn_off_visualization()
