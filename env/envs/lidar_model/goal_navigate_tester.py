import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model
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
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_normal, Stochastic_action_planner_uniform_bin, Stochastic_action_planner_uniform_bin_w_time_correlation, Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal
from raisimGymTorch.env.envs.lidar_model.action import Zeroth_action_planner, Modified_zeroth_action_planner
from raisimGymTorch.env.envs.lidar_model.storage import Buffer

"""
Check!!!!

1. action_planner type & params
2. collision_threshold
"""


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

np.random.seed(1)

# task specification
task_name = "point_goal_nav"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='trained environment model weight path', type=str, required=True)
parser.add_argument('-tw', '--tracking_weight', help='trained command tracking policy weight path', type=str, required=True)
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

assert cfg["environment"]["determine_env"] in [1, 2, 3], "Unavailable env type"
assert cfg["environment"]["evaluate"], "Change cfg[environment][evaluate] to True"
assert not cfg["environment"]["random_initialize"], "Change cfg[environment][random_initialize] to False"
assert cfg["environment"]["point_goal_initialize"] or cfg["environment"]["CVAE_data_collection_initialize"], "Change cfg[environment][point_goal_initialize] or cfg[environment][CVAE_data_collection_initialize] to True"
assert not cfg["environment"]["safe_control_initialize"], "Change cfg[environment][safe_control_initialize] to False"
assert not cfg["environment"]["CVAE_environment_initialize"], "Change cfg[environment][CVAE_environment_evaluation_initialize] to False"

# user command sampling
user_command = UserCommand(cfg, cfg['evaluating']['number_of_sample'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

# create environment from the configuration file
env = VecEnv(lidar_model.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Evaluating
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['data_collection']['command_period'] / cfg['environment']['control_dt'])
# evaluate_command_sampling_steps = math.floor(cfg['evaluating']['command_period'] / cfg['environment']['control_dt'])

state_dim = cfg["architecture"]["state_encoder"]["input"]
command_dim = cfg["architecture"]["command_encoder"]["input"]
P_col_dim = cfg["architecture"]["traj_predictor"]["collision"]["output"]
coordinate_dim = cfg["architecture"]["traj_predictor"]["coordinate"]["output"]   # Just predict x, y coordinate (not yaw)

use_TCN_COM_encoder = cfg["architecture"]["COM_encoder"]["use_TCN"]

if use_TCN_COM_encoder:
    # Use TCN for encoding COM vel history
    COM_feature_dim = cfg["architecture"]["COM_encoder"]["TCN"]["input"]
    COM_history_time_step = cfg["architecture"]["COM_encoder"]["TCN"]["time_step"]
    COM_history_update_period = int(cfg["architecture"]["COM_encoder"]["TCN"]["update_period"] / cfg["environment"]["control_dt"])
    assert state_dim - lidar_dim == cfg["architecture"]["COM_encoder"]["TCN"]["output"][-1], "Check COM_encoder output and state_encoder input in the cfg.yaml"
else:
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

iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

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

    # action_planner = Stochastic_action_planner_uniform_bin_w_time_correlation(command_range=cfg["environment"]["command"],
    #                                                                           n_sample=cfg["evaluating"]["number_of_sample"],
    #                                                                           n_horizon=n_prediction_step,
    #                                                                           n_bin=cfg["evaluating"]["number_of_bin"],
    #                                                                           beta=cfg["evaluating"]["beta"],
    #                                                                           gamma=cfg["evaluating"]["gamma"],
    #                                                                           noise_sigma=0.1,
    #                                                                           time_correlation_beta=cfg["evaluating"]["time_correlation_beta"],
    #                                                                           noise=False,
    #                                                                           action_dim=command_dim,
    #                                                                           random_command_sampler=user_command)
#
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

    # action_planner = Zeroth_action_planner(command_range=cfg["environment"]["command"],
    #                                        n_sample=cfg["evaluating"]["number_of_sample"],
    #                                        n_horizon=n_prediction_step,
    #                                        sigma=0.3,
    #                                        gamma=cfg["evaluating"]["gamma"],
    #                                        beta=0.6,
    #                                        action_dim=3)
#
    env.initialize_n_step()
    action_planner.reset()
    goal_position = env.set_goal()[np.newaxis, :]
    env.turn_on_visualization()
    COM_buffer.reset()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + "lidar_2d_normal_sampling" + '.mp4')

    # command tracking logging initialize
    command_traj = []

    # Initialize number of steps
    step = 0
    n_test_case = 0
    n_success_test_case = 0
    num_goals = 8

    # MUST safe period from collision
    MUST_safety_period = 3.0
    # MUST_safety_period = 2.0
    MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
    sample_user_command = np.zeros(3)
    prev_coordinate_obs = np.zeros((1, 3))

    pdb.set_trace()

    eval_start = time.time()

    collision_threshold = 0.05
    # collision_threshold = 0.8
    goal_distance_threshold = 10

    goal_time_limit = 180.
    goal_current_duration = 0.

    command_log = []

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

            if use_TCN_COM_encoder:
                COM_history_feature = COM_buffer.return_data()[0, :, :]
                COM_history_feature = np.tile(COM_history_feature, (cfg["evaluating"]["number_of_sample"], 1, 1))
                COM_history_feature = COM_history_feature.astype(np.float32)
                lidar_data = np.tile(lidar_data, (cfg["evaluating"]["number_of_sample"], 1))
                lidar_data = lidar_data.astype(np.float32)
                predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(COM_history_feature).to(device),
                                                                                   torch.from_numpy(lidar_data).to(device),
                                                                                   torch.from_numpy(action_candidates).to(device),
                                                                                   training=False)
            else:
                COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
                state = np.tile(np.concatenate((lidar_data, COM_history_feature)), (cfg["evaluating"]["number_of_sample"], 1))
                state = state.astype(np.float32)
                predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                   torch.from_numpy(action_candidates).to(device),
                                                                                   training=False)

            predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)

            # # Test
            # predicted_P_cols_label = np.where(predicted_P_cols > collision_threshold, 1, 0)
            # for sample_id in range(2000):
            #     total_collision_idx = np.argwhere(predicted_P_cols_label[:, sample_id])
            #     if len(total_collision_idx) != 0:
            #         first_collision_idx = np.min(total_collision_idx)
            #         predicted_coordinates[first_collision_idx+1:, sample_id, :] = np.tile(predicted_coordinates[first_collision_idx, sample_id, :], (12 - first_collision_idx - 1, 1))

            # compute reward (goal reward + safety reward)
            goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
            current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
            if current_goal_distance > goal_distance_threshold:
                goal_position_L *= (goal_distance_threshold / current_goal_distance)
            delta_goal_distance = current_goal_distance - np.sqrt(np.sum(np.power(predicted_coordinates - goal_position_L, 2), axis=-1))

            goal_reward = np.sum(delta_goal_distance, axis=0)
            goal_reward -= np.min(goal_reward)
            goal_reward /= np.max(goal_reward) + 1e-5  # normalize reward

            safety_reward = 1 - predicted_P_cols
            safety_reward = np.mean(safety_reward, axis=0)
            safety_reward /= np.max(safety_reward) + 1e-5  # normalize reward

            # command_difference_reward = - np.sqrt(np.sum(np.power(sample_user_command - action_candidates[0, :, :], 2), axis=-1))
            # command_difference_reward /= np.abs(np.min(command_difference_reward)) + 1e-5
            # command_difference_reward -= np.min(command_difference_reward)  # normalize reward

            action_size = np.sqrt((action_candidates[0, :, 0] / 1) ** 2 + (action_candidates[0, :, 1] / 0.4) ** 2 + (action_candidates[0, :, 2] / 1.2) ** 2)
            action_size /= np.max(action_size)

            reward = 1.0 * goal_reward * safety_reward + 0.3 * safety_reward
            # reward = 1.0 * goal_reward + 0.5 * safety_reward + 0.3 * action_size  # weighted sum for computing rewards
            # reward = 1.0 * goal_reward + 0.5 * safety_reward  # weighted sum for computing rewards
            coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]

            if len(coll_idx) != cfg["evaluating"]["number_of_sample"]:
                reward[coll_idx] = 0  # exclude trajectory that collides with obstacle

            cand_sample_user_command, sample_user_command_traj = action_planner.action(reward)

            # # plot predicted trajectory
            # traj_len, n_sample, coor_dim = predicted_coordinates.shape
            # for j in range(n_sample):
            #     plt.plot(predicted_coordinates[:, j, 0], predicted_coordinates[:, j, 1])
            # plt.savefig("sampled_traj (ours).png")
            # plt.clf()
            # pdb.set_trace()

        if new_action_time:
            sample_user_command = cand_sample_user_command.copy()

            if use_TCN_COM_encoder:
                COM_history_feature = COM_history_feature[0, :, :][np.newaxis, :]
                lidar_data = lidar_data[0, :][np.newaxis, :]
                predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(COM_history_feature).to(device),
                                                                                   torch.from_numpy(lidar_data).to(device),
                                                                                   torch.from_numpy(sample_user_command_traj[:, np.newaxis, :]).to(device),
                                                                                   training=False)
            else:
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

            # # reset action planner if stuck in local optimum
            # current_pos_change = np.sqrt(np.sum(np.power(init_coordinate_obs[0, :2] - prev_coordinate_obs[0, :2], 2)))
            # if current_pos_change < 0.005:  # 0.1 [m / s]
            #     action_planner.reset()

            prev_coordinate_obs = init_coordinate_obs.copy()

        command_log.append(sample_user_command)
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
        #     time_check.append(frame_end - frame_start)
        #     if len(time_check) == 500:
        #         time_check = np.array(time_check)
        #         print(f"Mean: {np.mean(time_check[50:])}")
        #         print(f"Std: {np.std(time_check[50:])}")
        #         pdb.set_trace()

        if new_action_time:
            print(frame_end - frame_start)

        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

        if wait_time > 0.:
            goal_current_duration += cfg['environment']['control_dt']
        else:
            goal_current_duration += (frame_end - frame_start)

        if goal_current_duration > goal_time_limit:
            done[0] = True

        # fail
        if done[0] == True:
            env.initialize_n_step()   # keep start in different initial condiition
            env.reset()
            COM_buffer.reset()
            action_planner.reset()
            goal_position = env.set_goal()[np.newaxis, :]
            n_test_case += 1
            step = 0
            command_traj = []
            sample_user_command = np.zeros(3)
            goal_current_duration = 0.
            print(f"Intermediate result : {n_success_test_case} / {n_test_case}")
        # success
        elif current_goal_distance < 0.5:
            # print(goal_current_duration)
            env.initialize_n_step()  # keep start in different initial condiition
            env.reset()
            # plot command trajectory
            command_traj = np.array(command_traj)
            # reset action planner and set new goal
            action_planner.reset()
            goal_position = env.set_goal()[np.newaxis, :]
            n_test_case += 1
            command_traj = []
            sample_user_command = np.zeros(3)
            n_success_test_case += 1
            goal_current_duration = 0.
            print(f"Intermediate result : {n_success_test_case} / {n_test_case} || Number of steps: {step}")
            step = 0

            plot_command_result(command_traj=np.array(command_log),
                                folder_name="command_trajectory",
                                task_name=task_name,
                                run_name="normal_fixed",
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

