from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_enviroment_model_param, UserCommand
from raisimGymTorch.helper.utils_plot import check_saving_folder
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
import argparse
import pdb
from raisimGymTorch.env.envs.lidar_model.model import Lidar_environment_model
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_uniform_bin, Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal
from raisimGymTorch.env.envs.lidar_model.storage import Buffer

"""
Must check 
(1) robot initialize position (should not be same throughout the env!) 
(2) goal position (should not be same throughout the env!)
(3) in all types of env (cylinder, box, corridor)
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


np.random.seed(0)

# task specification
task_name = "CVAE_data_collection"

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

assert not cfg["environment"]["evaluate"], "Change cfg[environment][evaluate] to False"
assert not cfg["environment"]["random_initialize"], "Change cfg[environment][random_initialize] to False"
assert not cfg["environment"]["point_goal_initialize"], "Change cfg[environment][point_goal_initialize] to False"
assert cfg["environment"]["CVAE_data_collection_initialize"], "Change cfg[environment][ CVAE_data_collection_initialize] to True"
assert not cfg["environment"]["safe_control_initialize"], "Change cfg[environment][safe_control_initialize] to False"

# user command sampling
user_command = UserCommand(cfg, cfg['evaluating']['number_of_sample'])

# create single environment from the configuration file
cfg['environment']['num_envs'] = 1

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

# iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
# weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    raise ValueError("Can't find trained weight, please provide a trained weight with --weight switch")

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

num_max_env = 360
num_max_sucess_goals_in_one_env = 8   # Should also change 'total_n_point_goal' in Environment.hpp if you change this value
num_max_data_in_one_goal = 20
print(f">> Check important data collection parameters: num_max_env = {num_max_env} / num_max_sucess_goals_in_one_env = {num_max_sucess_goals_in_one_env} / num_max_data_in_one_goal = {num_max_data_in_one_goal}")
pdb.set_trace()

# MUST safe period from collision
MUST_safety_period = 3.0
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['data_collection']['command_period'])
sample_user_command = np.zeros(3)
prev_coordinate_obs = np.zeros((1, 3))
goal_time_limit = 45.   # too filter trajectories trapped in local optima

# IMPORTANT PARAMETERS
collision_threshold = 0.05
goal_distance_threshold = 10   # Should change goal_set distance condition in Environment.hpp if you change this value (current: 10 for env1 & anv2, 5 for env3
print(f">> Check important action parameters: MUST_safety_period = {MUST_safety_period} / collision_threshold = {collision_threshold} / goal_distance_threshold = {goal_distance_threshold}")
pdb.set_trace()

# Monitor total saved data size
saved_data_size = {'env1': 0, 'env2': 0, 'env3': 0}

# Make the folder to save recorded data
folder_name = "CVAE_data"
check_saving_folder(folder_name)

for env_type in [1, 2, 3]:
    # Set environment type
    # 1: cylinder
    # 2: box
    # 3: corridor
    cfg["environment"]["determine_env"] = env_type
    print("==============================================")
    print(f"Environment {env_type} data collection started")
    for i in range(num_max_env):   # Change range value if you want to add more data continuing from previous data collection (ex) range(n) ==> range(n, n'))
        print(f"{i+1} / {num_max_env} ==>", end=' ')
        env_start = time.time()

        # Create environment from the configuration file
        cfg["environment"]["seed"]["train"] = env_type * 10000 + 2000 + i   # used seed: 12000 ~ 12000 + (num_max_env-1) | 22000 ~ 22000 + (num_max_env-1) | 32000 ~ 32000 + (num_max_env-1)
        env = VecEnv(lidar_model.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)
        env.load_scaling(command_tracking_weight_dir, int(iteration_number))
        # env.turn_on_visualization()

        current_num_success_goals = 0
        current_num_fail_goals = 0
        current_num_success_but_short_goals = 0

        # Count number of saved data in single environment
        # (maximum "num_max_sucess_goals_in_one_env" * "num_max_data_in_one_goal")
        data_count = 0

        while current_num_success_goals < num_max_sucess_goals_in_one_env:
            env.initialize_n_step()
            env.reset()
            action_planner.reset()
            goal_position = env.set_goal()[np.newaxis, :]
            COM_buffer.reset()

            # intialize counting variables
            goal_current_duration = 0.
            step = 0
            sample_user_command = np.zeros(3)

            # initialize data container
            observation_traj = []
            goal_position_traj = []
            command_traj = []

            if (current_num_fail_goals > 10) or (current_num_success_but_short_goals > 10):
                print("Not appropriate environment sampled")
                break

            while True:
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

                    predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)

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

                    # action_size = np.sqrt((action_candidates[0, :, 0] / 1) ** 2 + (action_candidates[0, :, 1] / 0.4) ** 2 + (action_candidates[0, :, 2] / 1.2) ** 2)
                    # action_size /= np.max(action_size)

                    reward = 1.0 * goal_reward * safety_reward + 0.3 * safety_reward
                    coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]

                    if len(coll_idx) != cfg["evaluating"]["number_of_sample"]:
                        reward[coll_idx] = 0  # exclude trajectory that collides with obstacle

                    cand_sample_user_command, sample_user_command_traj = action_planner.action(reward)

                    sample_user_command = cand_sample_user_command.copy()

                    # Visualize predicted modified command trajectory
                    state = state[0, :][np.newaxis, :]
                    predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                       torch.from_numpy(sample_user_command_traj[:, np.newaxis, :]).to(device),
                                                                                       training=False)
                    w_coordinate_modified_command_path = transform_coordinate_LW(init_coordinate_obs, predicted_coordinates[:, 0, :])
                    P_col_modified_command_path = predicted_P_cols[:, 0, :]
                    env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                        P_col_modified_command_path,
                                                        collision_threshold)
                    prev_coordinate_obs = init_coordinate_obs.copy()

                    # Record data needed for training CVAE
                    observation_traj.append(state[0].astype(np.float32))  # (state_dim,)
                    goal_position_traj.append(goal_position_L[0].astype(np.float32))  # (goal_position_L_dim,)
                    command_traj.append(sample_user_command.astype(np.float32))  # (command_dim,)

                tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
                tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
                tracking_obs = tracking_obs.astype(np.float32)

                with torch.no_grad():
                    tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))

                _, done = env.step(tracking_action.cpu().detach().numpy())

                step += 1

                frame_end = time.time()
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
                    # Will not record failure case
                    current_num_fail_goals += 1
                    break
                # success
                elif current_goal_distance < 0.5:
                    # Will record (sampled) success case
                    n_steps = len(observation_traj)
                    n_max_available_steps = n_steps - n_prediction_step
                    if n_max_available_steps >= num_max_data_in_one_goal:
                        sample_ids = np.random.choice(n_max_available_steps, num_max_data_in_one_goal, replace=False)
                        for sample_id in sample_ids:
                            data_count += 1

                            data_observation = observation_traj[sample_id]  # (observation_dim,)
                            data_goal_position = goal_position_traj[sample_id]  # (goal_position_dim,)
                            data_command_traj = np.stack(command_traj[sample_id:sample_id + n_prediction_step])  # (traj_len, command_dim)

                            # Save data
                            file_name = f"{env_type}_{i+1}_{data_count}"
                            np.savez_compressed(f"{folder_name}/{file_name}", observation=data_observation, goal_position=data_goal_position, command_traj=data_command_traj)

                        current_num_success_goals += 1
                    else:
                        current_num_success_but_short_goals += 1
                    break

        env_end = time.time()

        # Print elapse time and success rate to monitor data collection progress
        elapse_time_seconds = env_end - env_start
        elaspe_time_minutes = int(elapse_time_seconds / 60)
        elapse_time_seconds -= (elaspe_time_minutes * 60)
        elapse_time_seconds = int(elapse_time_seconds)
        saved_data_size[f'env{env_type}'] += (current_num_success_goals * num_max_data_in_one_goal)
        print(f"Time: {elaspe_time_minutes}m {elapse_time_seconds}s", "||" , f"Date SR: {num_max_sucess_goals_in_one_env} / {num_max_sucess_goals_in_one_env + current_num_fail_goals + current_num_success_but_short_goals} ({current_num_success_but_short_goals})",
            "||", f"Dataset: {saved_data_size['env1']} / {saved_data_size['env2']} / {saved_data_size['env3']}", sep=" ")

# env.stop_video_recording()
env.turn_off_visualization()

