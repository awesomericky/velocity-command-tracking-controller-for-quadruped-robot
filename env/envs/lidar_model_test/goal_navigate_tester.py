import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_test
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
from raisimGymTorch.env.envs.lidar_model.action import Zeroth_action_planner, Modified_zeroth_action_planner, Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal
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
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# task specification
task_name = "point_goal_nav_test"

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
env = VecEnv(lidar_model_test.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

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

# Use naive concatenation for encoding COM vel history
COM_feature_dim = cfg["architecture"]["COM_encoder"]["naive"]["input"]
COM_history_time_step = cfg["architecture"]["COM_encoder"]["naive"]["time_step"]
COM_history_update_period = int(cfg["architecture"]["COM_encoder"]["naive"]["update_period"] / cfg["environment"]["control_dt"])
assert state_dim - lidar_dim == COM_feature_dim * COM_history_time_step, "Check COM_encoder output and state_encoder input in the cfg.yaml"

command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

COM_buffer = Buffer(env.num_envs, COM_history_time_step, COM_feature_dim)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()

    # Load learned environment model weight
    loaded_environment_model = Lidar_environment_model(COM_encoding_config=cfg["architecture"]["COM_encoder"],
                                                       state_encoding_config=cfg["architecture"]["state_encoder"],
                                                       command_encoding_config=cfg["architecture"]["command_encoder"],
                                                       recurrence_config=cfg["architecture"]["recurrence"],
                                                       prediction_config=cfg["architecture"]["traj_predictor"],
                                                       device=device)
    loaded_environment_model.load_state_dict(torch.load(weight_path, map_location=device)['model_architecture_state_dict'])
    loaded_environment_model.eval()
    loaded_environment_model.to(device)

    # Load action planner
    n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
    # action_planner = Stochastic_action_planner_uniform_bin(command_range=cfg["environment"]["command"],
    #                                                        n_sample=cfg["evaluating"]["number_of_sample"],
    #                                                        n_horizon=n_prediction_step,
    #                                                        n_bin=cfg["evaluating"]["number_of_bin"],
    #                                                        beta=cfg["evaluating"]["beta"],
    #                                                        gamma=cfg["evaluating"]["gamma"],
    #                                                        noise_sigma=0.1,
    #                                                        noise=False,
    #                                                        action_dim=command_dim)
    action_planner = Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal(command_range=cfg["environment"]["command"],
                                                                                     n_sample=cfg["evaluating"]["number_of_sample"],
                                                                                     n_horizon=n_prediction_step,
                                                                                     n_bin=cfg["evaluating"]["number_of_bin"],
                                                                                     beta=cfg["evaluating"]["beta"],
                                                                                     gamma=cfg["evaluating"]["gamma"],
                                                                                     sigma=cfg["evaluating"]["sigma"],
                                                                                     noise_sigma=0.1,
                                                                                     noise=False,
                                                                                     action_dim=command_dim,
                                                                                     random_command_sampler=user_command)

    env.initialize_n_step()
    env.reset()
    action_planner.reset()
    goal_position = env.set_goal()[np.newaxis, :]
    env.turn_on_visualization()
    COM_buffer.reset()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + "lidar_2d_normal_sampling" + '.mp4')

    # Initialize number of steps
    step = 0
    n_test_case = 0
    if cfg["environment"]["type"] == 2:
        num_goals = 3
    else:
        num_goals = 1

    # MUST safe period from collision
    MUST_safety_period = 3.0
    MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
    sample_user_command = np.zeros(3)

    # Needed for computing real time factor
    total_time = 0
    total_n_step = 0

    collision_threshold = 0.05
    goal_distance_threshold = 10

    # collision idx list initialize
    num_collision_idx = []

    # command tracking logging initialize
    command_log = []

    pdb.set_trace()

    while n_test_case < num_goals:
        frame_start = time.time()
        new_action_time = step % command_period_steps == 0

        obs, _ = env.observe(False)  # observation before taking step
        if step % COM_history_update_period == 0:
            COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
            COM_buffer.update(COM_feature)

        if new_action_time:
            lidar_data = obs[0, proprioceptive_sensor_dim:]
            action_candidates = action_planner.sample()
            action_candidates = np.swapaxes(action_candidates, 0, 1)
            action_candidates = action_candidates.astype(np.float32)
            init_coordinate_obs = env.coordinate_observe()

            COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
            state = np.tile(np.concatenate((lidar_data, COM_history_feature)), (cfg["evaluating"]["number_of_sample"], 1))
            state = state.astype(np.float32)
            predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                               torch.from_numpy(action_candidates).to(device),
                                                                               training=False)

            predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)  # (12, 2000)

            # compute reward (goal reward + safety reward)
            goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
            current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
            if current_goal_distance > goal_distance_threshold:
                goal_position_L *= (goal_distance_threshold / current_goal_distance)
            current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
            delta_goal_distance = current_goal_distance - np.sqrt(np.sum(np.power(predicted_coordinates - goal_position_L, 2), axis=-1))
            # delta_goal_distance *= goal_kernel

            goal_reward = np.sum(delta_goal_distance, axis=0)
            goal_reward -= np.min(goal_reward)
            goal_reward /= np.max(goal_reward) + 1e-5  # normalize reward

            # goal_reward = np.sum(delta_goal_distance / current_goal_distance, axis=0)
            # goal_reward /= np.max(goal_reward) + 1e-5

            safety_reward = 1 - predicted_P_cols
            safety_reward = np.mean(safety_reward, axis=0)
            safety_reward /= np.max(safety_reward) + 1e-5  # normalize reward

            # command_difference_reward = - np.sqrt(np.sum(np.power(sample_user_command - action_candidates[0, :, :], 2), axis=-1))
            # command_difference_reward /= np.abs(np.min(command_difference_reward)) + 1e-5
            # command_difference_reward -= np.min(command_difference_reward)  # normalize reward

            # action_size = np.sqrt((action_candidates[0, :, 0] / 1) ** 2 + (action_candidates[0, :, 1] / 0.4) ** 2 + (action_candidates[0, :, 2] / 1.2) ** 2)
            # action_size /= np.max(action_size)
            # action_size = abs(action_candidates[0, :, 0] / 1) + abs(action_candidates[0, :, 1] / 0.4) + abs(action_candidates[0, :, 2] / 1.2)
            # action_size /= 3

            reward = 1.0 * goal_reward * safety_reward + 0.3 * safety_reward
            # reward = 1.0 * goal_reward + 0.5 * safety_reward + 0.3 * action_size  # final reward term for (env1, 3, 4)
            # reward = 1.4 * goal_reward + 0.7 * safety_reward + 0.4 * action_size  # final reward term for (env1, 3, 4)
            coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]

            if len(coll_idx) != cfg["evaluating"]["number_of_sample"]:
                reward[coll_idx] = 0  # exclude trajectory that collides with obstacle

            # plt.hist(reward)
            # plt.savefig("reward_hist")
            # plt.clf()
            # print(len(coll_idx))
            # pdb.set_trace()

            # if total_time > 16:
            #     # goal_reward[coll_idx] = 0
            #     # safety_reward[coll_idx] = 0
            #     # command_difference_reward[coll_idx] = 0
            #     labels = np.arange(cfg["evaluating"]["number_of_sample"])
            #     plt.bar(labels, 2.3 * goal_reward, width=10, label='goal')
            #     plt.bar(labels, 3.9 * safety_reward, width=10, label='safety', bottom=2.3 * goal_reward)
            #     plt.bar(labels, 0.1 * command_difference_reward, width=10, label='command difference', bottom=2.3 * goal_reward + 3.9 * safety_reward)
            #     plt.legend()
            #     plt.savefig("reward_log")
            #     plt.clf()
            #     pdb.set_trace()

            cand_sample_user_command, sample_user_command_traj = action_planner.action(reward)
            sample_user_command = cand_sample_user_command.copy()

        # if step % 200 == 0:
        #     # plot predicted trajectory
        #     traj_len, n_sample, coor_dim = predicted_coordinates.shape
        #     for j in range(n_sample):
        #         plt.plot(predicted_coordinates[:, j, 0], predicted_coordinates[:, j, 1])
        #     plt.title("Sampled trajectory (n_sample: 2000)")
        #     plt.savefig("sampled_traj_all.png")
        #     plt.clf()

        #
        #     # plot predicted trajectory
        #     traj_len, n_sample, coor_dim = predicted_coordinates.shape
        #     modified_predicted_coordinates = np.concatenate((np.zeros((1, n_sample, coor_dim)), predicted_coordinates), axis=0)
        #     for i in range(n_sample):
        #         if i not in coll_idx:
        #             # plot_label = "{:.2f}, {:.2f}, {:.2f}".format(action_candidates[i, 0], action_candidates[i, 1], action_candidates[i, 2])
        #             # plt.plot(future_position[i, :, 0], future_position[i, :, 1], label=plot_label)
        #             plt.plot(modified_predicted_coordinates[:, i, 0], modified_predicted_coordinates[:, i, 1])
        #     plt.title("Sampled trajectory (n_sample: 2000)")
        #     plt.savefig("sampled_traj_safe.png")
        #     plt.clf()
        #
        #     print(len(coll_idx))

            # predict modified command trajectory
            state = state[0, :][np.newaxis, :]
            predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                               torch.from_numpy(sample_user_command_traj[:, np.newaxis, :]).to(device),
                                                                               training=False)

            # visualize predicted modified command trajectory
            w_coordinate_modified_command_path = transform_coordinate_LW(init_coordinate_obs, predicted_coordinates[:, 0, :])
            P_col_modified_command_path = predicted_P_cols[:, 0, :]
            env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                P_col_modified_command_path,
                                                collision_threshold)

        # Command logging
        command_log.append(sample_user_command)

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
            # COM_buffer.reset()
            # action_planner.reset()
            # sample_user_command = np.zeros(3)
            # step = 0

        frame_end = time.time()

        # # (2000 sample, 10 bin ==> 0.008 sec)
        # print(frame_end - frame_start)
        # if new_action_time:
        #     print(f"Action: {frame_end - frame_start}")

        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

        if wait_time > 0:
            total_time += cfg['environment']['control_dt']
        else:
            total_time += (frame_end - frame_start)
        total_n_step += 1

        if current_goal_distance < 0.5:
            # pdb.set_trace()
            # plot command trajectory
            command_log = np.array(command_log)
            plot_command_result(command_traj=command_log,
                                folder_name="command_trajectory",
                                task_name=task_name,
                                run_name="normal_fixed",
                                n_update=n_test_case,
                                control_dt=cfg["environment"]["control_dt"])

            # reset action planner and set new goal
            action_planner.reset()
            goal_position = env.set_goal()[np.newaxis, :]
            n_test_case += 1
            step = 0
            command_log = []
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
