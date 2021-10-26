from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_test
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
from raisimGymTorch.env.envs.lidar_model.action import Modified_action_planner
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
from fastdtw import fastdtw


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


# task specification
task_name = "lidar_environment_model_test"

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
evaluate_command_sampling_steps = math.floor(cfg['evaluating']['command_period'] / cfg['environment']['control_dt'])

state_dim = cfg["architecture"]["state_encoder"]["input"]
command_dim = cfg["architecture"]["command_encoder"]["input"]
P_col_dim = cfg["architecture"]["traj_predictor"]["collision"]["output"]
coordinate_dim = cfg["architecture"]["traj_predictor"]["coordinate"]["output"]   # Just predict x, y coordinate (not yaw)

assert cfg["architecture"]["COM_encoder"]["use_TCN"] == False, "TCN not yet included in lidar environment model"
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
    loaded_environment_model = Lidar_environment_model(lidar_encoding_config=cfg["architecture"]["state_encoder"],
                                                       command_encoding_config=cfg["architecture"]["command_encoder"],
                                                       recurrence_config=cfg["architecture"]["recurrence"],
                                                       prediction_config=cfg["architecture"]["traj_predictor"],
                                                       device=device)
    loaded_environment_model.load_state_dict(torch.load(weight_path)['model_architecture_state_dict'])
    loaded_environment_model.eval()
    loaded_environment_model.to(device)

    # Load action planner
    n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
    action_planer = Modified_action_planner(command_range=cfg["environment"]["command"],
                                            n_sample=cfg["evaluating"]["n_sample"],
                                            n_horizon=n_prediction_step,
                                            sigma=cfg["evaluating"]["sigma"],
                                            beta=cfg["evaluating"]["beta"],
                                            action_dim=command_dim)

    env.initialize_n_step()
    env.reset()
    env.turn_on_visualization()
    COM_buffer.reset()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

    # command tracking logging initialize
    desired_command_traj = []
    modified_command_traj = []

    # max_steps = 1000000
    max_steps = 3000 ## 30 secs

    for step in range(max_steps):
        frame_start = time.time()
        new_action_time = step % command_period_steps == 0
        new_command_sampling_time = step % evaluate_command_sampling_steps == 0

        if new_command_sampling_time:
            desired_command = user_command.uniform_sample_evaluate()[0, :]
            desired_command[0] = 1
            desired_command[1] = 0
            desired_command[2] = 0
            sample_user_command = desired_command

        obs, _ = env.observe(False)  # observation before taking step
        if step % COM_history_update_period == 0:
            COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
            COM_buffer.update(COM_feature)

        if new_action_time:
            lidar_data = obs[0, proprioceptive_sensor_dim:]
            COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
            state = np.tile(np.concatenate((lidar_data, COM_history_feature)), (cfg["evaluating"]["n_sample"], 1))
            action_candidates = action_planer.sample(desired_command)
            action_candidates = np.swapaxes(action_candidates, 0, 1)
            init_coordinate_obs = env.coordinate_observe()

            state = state.astype(np.float32)
            action_candidates = action_candidates.astype(np.float32)

            predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                               torch.from_numpy(action_candidates).to(device),
                                                                               training=False)
            desired_command_path = predicted_coordinates[:, 0, :]
            predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
            reward = np.zeros(cfg["evaluating"]["n_sample"])
            distance = np.zeros(cfg["evaluating"]["n_sample"])

            # visualize predicted desired command trajectory
            w_coordinate_desired_command_path = transform_coordinate(init_coordinate_obs, desired_command_path)
            P_col_desired_command_path = predicted_P_cols[:, 0]
            env.visualize_desired_command_traj(w_coordinate_desired_command_path,
                                               P_col_desired_command_path)

            if len(np.where(predicted_P_cols[:, 0] > 0.5)[0]) == 0:
                # current desired command is safe
                sample_user_command = action_planer.action(reward, safe=True)
            else:
                # current desired command is not safe
                reward = np.exp(-3 * np.sum(predicted_P_cols, axis=0))
                for i in range(cfg["evaluating"]["n_sample"]):
                    distance[i], _ = fastdtw(desired_command_path, predicted_coordinates[:, i, :])
                distance = np.exp(- distance / n_prediction_step)
                reward += distance
                sample_user_command, env_idx = action_planer.action(reward)

                # visualize predicted modified command trajectory
                w_coordinate_modified_command_path = transform_coordinate(init_coordinate_obs, predicted_coordinates[:, env_idx, :])
                P_col_modified_command_path = predicted_P_cols[:, env_idx]
                env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                    P_col_modified_command_path)

        tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
        tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
        tracking_obs = tracking_obs.astype(np.float32)

        with torch.no_grad():
            tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
        _, done = env.step(tracking_action.cpu().detach().numpy())

        # reset COM buffer for terminated environment
        if done[0] == True:
            env.reset()
            COM_buffer.reset()

        # Command logging
        desired_command_traj.append(desired_command)
        modified_command_traj.append(sample_user_command)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

    # env.stop_video_recording()
    env.turn_off_visualization()

    desired_command_traj = np.array(desired_command_traj)
    modified_command_traj = np.array(modified_command_traj)
    plot_command_tracking_result(desired_command_traj, modified_command_traj,
                                  weight_path.split('/')[-3], weight_path.split('/')[-2],
                                  "test", control_dt=cfg['environment']['control_dt'])