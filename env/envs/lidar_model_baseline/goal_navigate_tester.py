import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_baseline
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

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# task specification
task_name = "lidar_model_baseline_CWM"

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
env = VecEnv(lidar_model_baseline.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Evaluating
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['data_collection']['command_period'] / cfg['environment']['control_dt'])

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
action_planner = Stochastic_action_planner_uniform_bin_baseline(command_range=cfg["environment"]["command"],
                                                                 n_sample=cfg["evaluating"]["number_of_sample"],
                                                                 n_horizon=n_prediction_step,
                                                                 n_bin=cfg["evaluating"]["number_of_bin"],
                                                                 beta=cfg["evaluating"]["beta"],
                                                                 gamma=cfg["evaluating"]["gamma"],
                                                                 action_dim=command_dim)


#action_planner = Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal(command_range=cfg["environment"]["command"],
#                                                                                 n_sample=cfg["evaluating"]["number_of_sample"],
#                                                                                 n_horizon=n_prediction_step,
#                                                                                 n_bin=cfg["evaluating"]["number_of_bin"],
#                                                                                 beta=cfg["evaluating"]["beta"],
#                                                                                 gamma=cfg["evaluating"]["gamma"],
#                                                                                 sigma=cfg["evaluating"]["sigma"],
#                                                                                 noise_sigma=0.1,
#                                                                                 noise=False,
#                                                                                 action_dim=command_dim,
#                                                                                 random_command_sampler=user_command)

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
n_success_test_case = 0
num_goals = 8

goal_distance_threshold = 10

total_step = 0

# MUST safe period from collision
MUST_safety_period = 3.  # not 2 because we want long distance planning
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
sample_user_command = np.zeros(3)
prev_coordinate_obs = np.zeros((1, 3))
goal_rewards = np.zeros((cfg["evaluating"]["number_of_sample"], 1), dtype=np.float32)
collision_idx_list = np.zeros((cfg["evaluating"]["number_of_sample"], 1), dtype=np.float32)

pdb.set_trace()

eval_start = time.time()

local_optimum_start_time = 0

goal_time_limit = 180.
goal_current_duration = 0.
command_log = []

# log traversal distance
traversal_distance = 0.
previous_coordinate = None
current_coordinate = None

while n_test_case < num_goals:

    frame_start = time.time()
    new_action_time = step % command_period_steps == 0

    obs, _ = env.observe(False)  # observation before taking step

    init_coordinate_obs = env.coordinate_observe()

    if new_action_time:
        action_candidates = action_planner.sample()
        action_candidates = action_candidates.astype(np.float32)

        # log traversal distance (just env 0)
        previous_coordinate = current_coordinate
        current_coordinate = env.coordinate_observe()
        if previous_coordinate is not None:
            delta_coordinate = current_coordinate[0, :-1] - previous_coordinate[0, :-1]
            traversal_distance += (delta_coordinate[0] ** 2 + delta_coordinate[1] ** 2) ** 0.5

        # # predict trajectory
        # n_sample = action_candidates.shape[0]
        # n_prediction = 12
        # delta_t = 0.5
        # future_position = np.zeros((n_sample, n_prediction + 1, 2))
        # for i in range(action_candidates.shape[0]):
        #     local_yaw = 0
        #     local_x = 0
        #     local_y = 0
        #     vel_x, vel_y, vel_yaw = action_candidates[i, :]
        #     for j in range(n_prediction):
        #         local_yaw += vel_yaw * delta_t
        #         local_x += vel_x * delta_t * np.cos(local_yaw) - vel_y * delta_t * np.sin(local_yaw)
        #         local_y += vel_x * delta_t * np.sin(local_yaw) + vel_y * delta_t * np.cos(local_yaw)
        #         future_position[i, j+1, 0] = local_x
        #         future_position[i, j+1, 1] = local_y
        #
        # # plot predicted trajectory
        # n_sample, traj_len, coor_dim = future_position.shape
        # for i in range(n_sample):
        #     plt.plot(future_position[i, :, 0], future_position[i, :, 1])
        # plt.savefig("sampled_traj (baseline).png")
        # plt.clf()
        # pdb.set_trace()



        # compute reward (goal reward + safety reward)
        goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
        if current_goal_distance > goal_distance_threshold:
            goal_position_L *= (goal_distance_threshold / current_goal_distance)

        ##### Needed check
        # reward_compute_start = time.time()
        goal_rewards, collision_idx_list = env.baseline_compute_reward(action_candidates, np.swapaxes(goal_position_L, 0, 1), goal_rewards, collision_idx_list,
                                                                       n_prediction_step, cfg["data_collection"]["command_period"], MUST_safety_period)
        # reward_compute_end = time.time()
        coll_idx = np.where(collision_idx_list == 1)[0]
        if cfg["evaluating"]["number_of_sample"] > 1:
            goal_rewards -= np.min(goal_rewards)
            goal_rewards /= np.max(goal_rewards) + 1e-5
        else:
            command_difference_rewards = 0

        # action_size = np.sqrt((action_candidates[:, 0] / 1) ** 2 + (action_candidates[:, 1] / 0.4) ** 2 + (action_candidates[:, 2] / 1.2) ** 2)
        # action_size /= np.max(action_size)

        # reward = 1.2 * np.squeeze(goal_rewards, -1) + 0.1 * action_size
        reward = 1. * np.squeeze(goal_rewards, -1)

        if len(coll_idx) != cfg["evaluating"]["number_of_sample"]:
            reward[coll_idx] = 0  # exclude trajectory that collides with obstacle

        sample_user_command = action_planner.action(reward)

        # # reset action planner if stuck in local optimum
        # current_pos_change = np.sqrt(np.sum(np.power(init_coordinate_obs[0, :2] - prev_coordinate_obs[0, :2], 2)))
        # if current_pos_change < 0.005:  # 0.1 [m / s]
        #     if local_optimum_start_time == 0.0:
        #         local_optimum_start_time = time.time()
        #     action_planner.reset()

        prev_coordinate_obs = init_coordinate_obs.copy()

    command_log.append(sample_user_command.copy())
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

    # # (2000 sample, 10 bin ==> 0.008 sec)
    # print(frame_end - frame_start)
    # if new_action_time:
    #     time_check.append(frame_end - frame_start)
    #     if len(time_check) == 500:
    #         time_check = np.array(time_check)
    #         print(f"Mean: {np.mean(time_check[50:])}")
    #         print(f"Std: {np.std(time_check[50:])}")
    #         pdb.set_trace()

    # if new_action_time:
    #     print(frame_end - frame_start)

    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

    if wait_time > 0.:
        time.sleep(wait_time)

    if wait_time > 0.:
        goal_current_duration += cfg['environment']['control_dt']
    else:
        goal_current_duration += (frame_end - frame_start)

    total_step += 1

    if goal_current_duration > goal_time_limit:
        done[0] = True

    # fail
    if done[0] == True:
        env.reset()
        action_planner.reset()
        goal_position = env.set_goal()[np.newaxis, :]
        n_test_case += 1
        step = 0
        command_traj = []
        sample_user_command = np.zeros(3)
        goal_current_duration = 0.
        print(f"Intermediate result : {n_success_test_case} / {n_test_case}")
        total_step = 0
        traversal_distance = 0.
        previous_coordinate = None
        current_coordinate = None
    # success
    elif current_goal_distance < 0.5:
        env.reset()
        # plot command trajectory
        command_traj = np.array(command_traj)
        # reset action planner and set new goal
        action_planner.reset()
        goal_position = env.set_goal()[np.newaxis, :]
        n_test_case += 1
        step = 0
        command_traj = []
        sample_user_command = np.zeros(3)
        n_success_test_case += 1
        goal_current_duration = 0.
        print(f"Intermediate result : {n_success_test_case} / {n_test_case} || Total step: {total_step} || Traversal distance: {traversal_distance}")
        total_step = 0
        traversal_distance = 0.
        previous_coordinate = None
        current_coordinate = None

        plot_command_result(command_traj=np.array(command_log),
                            folder_name="command_trajectory",
                            task_name=task_name,
                            run_name="baseline",
                            n_update=n_test_case,
                            control_dt=cfg["environment"]["control_dt"])
        command_log = []

eval_end = time.time()

print("===========================================")
print(f"Result : {n_success_test_case} / {num_goals}")
print(f"Time: {eval_end - eval_start}")
print("===========================================")

# env.stop_video_recording()
env.turn_off_visualization()

