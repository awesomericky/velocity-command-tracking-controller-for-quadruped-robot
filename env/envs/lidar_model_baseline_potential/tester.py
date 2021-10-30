from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_enviroment_model_param, UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_tracking_result
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
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
from fastdtw import fastdtw
import matplotlib.pyplot as plt


def transform_coordinate(w_init_coordinate, l_coordinate_traj):
    """
    Transform LOCAL frame coordinate trajectory to WORLD frame coordinate trajectory

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param l_coordinate_traj: coordintate trajectory in LOCAL frame (n_prediction_step, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    w_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix)
    w_coordinate_traj += w_init_coordinate[:, :-1]
    return w_coordinate_traj

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

# config (load saved configuration)
# cfg = YAML().load(open(weight_path.rsplit("/", 1)[0] + "/cfg.yaml", 'r'))

# user command sampling
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

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
evaluate_command_sampling_steps = math.floor(cfg['evaluating']['command_period'] / cfg['environment']['control_dt'])

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
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path)['actor_architecture_state_dict'])
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
    loaded_environment_model.load_state_dict(torch.load(weight_path)['model_architecture_state_dict'])
    loaded_environment_model.eval()
    loaded_environment_model.to(device)

    # Load action planner
    n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
    action_planer = Stochastic_action_planner_normal(command_range=cfg["environment"]["command"],
                                                     n_sample=cfg["evaluating"]["number_of_sample"],
                                                     n_horizon=n_prediction_step,
                                                     sigma=cfg["evaluating"]["sigma"],
                                                     gamma=cfg["evaluating"]["gamma"],
                                                     noise_sigma=0.1,
                                                     noise=True,
                                                     action_dim=command_dim)

    env.initialize_n_step()
    env.reset()
    env.turn_on_visualization()
    COM_buffer.reset()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_" + "lidar_2d_normal_sampling" + '.mp4')

    # command tracking logging initialize
    desired_command_traj = []
    modified_command_traj = []

    # max_steps = 1000000
    num_test_case = 100
    current_test_case = 0
    failed_test_case = 0
    failed_case = []
    sampled_user_command_history = []

    # MUST safe period from collision
    MUST_safety_period = 3.5
    MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])

    pdb.set_trace()
    """
    # Without safety controller
    for i in range(num_test_case):
        desired_command = user_command.uniform_sample_evaluate()[0, :]
        sampled_user_command_history.append(desired_command.copy())
        sample_user_command = desired_command

        for step in range(evaluate_command_sampling_steps * 2):  # 6 [s]
            frame_start = time.time()
            new_action_time = step % command_period_steps == 0

            obs, _ = env.observe(False)  # observation before taking step
            if step % COM_history_update_period == 0:
                COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                COM_buffer.update(COM_feature)

            if new_action_time:
                lidar_data = obs[0, proprioceptive_sensor_dim:]
                action_candidates = action_planer.sample(desired_command)
                action_candidates = np.swapaxes(action_candidates, 0, 1)
                action_candidates = action_candidates.astype(np.float32)
                init_coordinate_obs = env.coordinate_observe()

                if use_TCN_COM_encoder:
                    COM_history_feature = COM_buffer.return_data()[0, :, :]
                    COM_history_feature = np.tile(COM_history_feature, (cfg["evaluating"]["number_of_sample"], 1))
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

                desired_command_path = predicted_coordinates[:, 0, :]
                predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                reward = np.zeros(cfg["evaluating"]["number_of_sample"])
                distance = np.zeros(cfg["evaluating"]["number_of_sample"])

                # visualize predicted desired command trajectory
                w_coordinate_desired_command_path = transform_coordinate(init_coordinate_obs, desired_command_path)
                P_col_desired_command_path = predicted_P_cols[:, 0]
                env.visualize_desired_command_traj(w_coordinate_desired_command_path,
                                                   P_col_desired_command_path)

            tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
            tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
            tracking_obs = tracking_obs.astype(np.float32)

            with torch.no_grad():
                tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
            _, done = env.step(tracking_action.cpu().detach().numpy())

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

            if wait_time > 0.:
                time.sleep(wait_time)

            # if new_action_time:
            #     print(frame_end - frame_start)

            # reset COM buffer for terminated environment
            if done[0] == True:
                failed_test_case += 1
                failed_case.append(i)
                env.initialize_n_step()
                env.reset()
                COM_buffer.reset()
                break

        # if predicted_collision:
        #     current_test_case += 1

    print(f"Result: {num_test_case - failed_test_case} / {num_test_case}")

    failed_case = np.array(failed_case)
    sampled_user_command_history = np.array(sampled_user_command_history)
    np.savez("safety_controller_failed_case", failure=failed_case)
    np.savez("command_history", command_history=sampled_user_command_history)

    """

    # With safety controller
    comparing_failed_case = np.load('safety_controller_failed_case.npz')['failure']
    sampled_user_command_history = np.load('command_history.npz')['command_history']

    for i in range(num_test_case):
        desired_command = sampled_user_command_history[i, :]
        sample_user_command = desired_command

        for step in range(evaluate_command_sampling_steps * 2):  # 6 [s]
            frame_start = time.time()
            new_action_time = step % command_period_steps == 0

            # predict_collision = False

            obs, _ = env.observe(False)  # observation before taking step
            if step % COM_history_update_period == 0:
                COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                COM_buffer.update(COM_feature)

            if new_action_time:
                lidar_data = obs[0, proprioceptive_sensor_dim:]
                action_candidates = action_planer.sample(desired_command)
                action_candidates = np.swapaxes(action_candidates, 0, 1)
                action_candidates = action_candidates.astype(np.float32)
                init_coordinate_obs = env.coordinate_observe()

                if use_TCN_COM_encoder:
                    COM_history_feature = COM_buffer.return_data()[0, :, :]
                    COM_history_feature = np.tile(COM_history_feature, (cfg["evaluating"]["number_of_sample"], 1))
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

                desired_command_path = predicted_coordinates[:, 0, :]
                predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                reward = np.zeros(cfg["evaluating"]["number_of_sample"])
                # distance = np.zeros(cfg["evaluating"]["number_of_sample"])

                # visualize predicted desired command trajectory
                w_coordinate_desired_command_path = transform_coordinate(init_coordinate_obs, desired_command_path)
                P_col_desired_command_path = predicted_P_cols[:, 0]
                env.visualize_desired_command_traj(w_coordinate_desired_command_path,
                                                   P_col_desired_command_path)

                if len(np.where(predicted_P_cols[:MUST_safety_period_n_steps, 0] > 0.5)[0]) == 0:
                    # current desired command is safe
                    sample_user_command = action_planer.action(reward, safe=True)
                else:
                    # predict_collision = True

                    # current desired command is not safe
                    reward = 3 * np.exp(-3 * np.sum(predicted_P_cols, axis=0))
                    # for i in range(cfg["evaluating"]["number_of_sample"]):
                    #     distance[i], _ = fastdtw(desired_command_path, predicted_coordinates[:, i, :])
                    # distance = np.exp(- distance / n_prediction_step)
                    # reward += 0.5 * distance

                    # labels = np.arange(cfg["evaluating"]["number_of_sample"])
                    # plt.bar(labels, 3 * np.exp(-3 * np.sum(predicted_P_cols, axis=0)), label='safety')
                    # plt.bar(labels, 0.5 * distance, label='nDTW', bottom=3 * np.exp(-3 * np.sum(predicted_P_cols, axis=0)))
                    # plt.legend()
                    # plt.savefig("reward_log")
                    # plt.clf()
                    # pdb.set_trace()

                    sample_user_command, sample_user_command_traj = action_planer.action(reward)

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
                    w_coordinate_modified_command_path = transform_coordinate(init_coordinate_obs, predicted_coordinates[:, 0, :])
                    P_col_modified_command_path = predicted_P_cols[:, 0, :]
                    env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                        P_col_modified_command_path)

            tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
            tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
            tracking_obs = tracking_obs.astype(np.float32)

            with torch.no_grad():
                tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
            _, done = env.step(tracking_action.cpu().detach().numpy())

            # Command logging
            desired_command_traj.append(desired_command)
            modified_command_traj.append(sample_user_command)

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

            if wait_time > 0.:
                time.sleep(wait_time)

            # if new_action_time and predict_collision:
            #     print(frame_end - frame_start)
            #     pdb.set_trace()

            # reset COM buffer for terminated environment
            if done[0] == True:
                failed_test_case += 1
                failed_case.append(i)
                env.initialize_n_step()
                env.reset()
                COM_buffer.reset()
                break

    print(f"Result: {num_test_case - failed_test_case} / {num_test_case}")

    # env.stop_video_recording()
    env.turn_off_visualization()

    # n_all_fail = 0
    # failed_case = np.array(failed_case)
    # for case in comparing_failed_case:
    #     if case in failed_case:
    #         n_all_fail += 1
    # n_only_wo_safety_fail = len(comparing_failed_case) - n_all_fail
    # n_only_w_safety_fail = len(failed_case) - n_all_fail
    # print(f"Only without safety fail: {n_only_wo_safety_fail}")
    # print(f"Only with safety fail: {n_only_w_safety_fail}")
    # print(f"All fail: {n_all_fail}")
    # print(f"Precentage of safety: {len(comparing_failed_case) - n_all_fail} / {len(comparing_failed_case)}")

    desired_command_traj = np.array(desired_command_traj)
    modified_command_traj = np.array(modified_command_traj)
    plot_command_tracking_result(desired_command_traj, modified_command_traj,
                                  weight_path.split('/')[-3], weight_path.split('/')[-2],
                                  "test", control_dt=cfg['environment']['control_dt'])
    # """
    

    