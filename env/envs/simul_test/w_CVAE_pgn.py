import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import simul_test
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_enviroment_model_param, UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_result, check_saving_folder
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
import pdb
from raisimGymTorch.env.envs.lidar_model.model import Lidar_environment_model
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_normal, Stochastic_action_planner_uniform_bin, Stochastic_action_planner_uniform_bin_w_time_correlation, Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal
from raisimGymTorch.env.envs.lidar_model.action import Zeroth_action_planner, Modified_zeroth_action_planner
from raisimGymTorch.env.envs.lidar_model.action import Stochastic_action_planner_w_CVAE
from raisimGymTorch.env.envs.lidar_model.model import CVAE_implicit_distribution_inference
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
import random
import json
from collections import defaultdict
import wandb
from shutil import copyfile

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

def compute_num_collision(collision_idx):
    """
    :param collision_idx: list of steps where collision occurred (list)
    :return: num_collsiion : number of collision (int)

    ex) collision_idx = [2, 3, 4, 10, 11, 15] ==> num_collision = 3
    """
    num_collision = 0
    for i in range(len(collision_idx) - 1):
        if collision_idx[i+1] - collision_idx[i] != 1:
            num_collision += 1

    if len(collision_idx) == 0:
        num_collision = 0
    else:
        num_collision += 1

    return num_collision

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# task specification
task_name = "Simple_point_goal_nav"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--validate', help='validation or test', type=bool, default=False)
parser.add_argument('-w', '--weight', help='trained environment model weight path', type=str, required=True)
parser.add_argument('-cw', '--cvae_weight', help='trained CVAE model weight path', type=str, required=True)
parser.add_argument('-tw', '--tracking_weight', help='trained command tracking policy weight path', type=str, required=True)
args = parser.parse_args()
validation = args.validate
weight_path = args.weight
cvae_weight_path = args.cvae_weight
command_tracking_weight_path = args.tracking_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

assert cfg["environment"]["test_initialize"]["point_goal"], "Change cfg[environment][test_initialize][point_goal] to True"
assert not cfg["environment"]["test_initialize"]["safety_control"], "Change cfg[environment][test_initialize][safety_control] to False"

# user command sampling
user_command = UserCommand(cfg, cfg['CVAE']['planner']['wo_CVAE_number_of_sample'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

# create environment from the configuration file
env = VecEnv(simul_test.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
state_dim = cfg["environment_model"]["architecture"]["state_encoder"]["input"]
command_period_steps = math.floor(cfg['command_tracking']['command_period'] / cfg['environment']['control_dt'])
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Use naive concatenation for encoding COM vel history
COM_feature_dim = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["input"]
COM_history_time_step = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["time_step"]
COM_history_update_period = int(cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["update_period"] / cfg["environment"]["control_dt"])
assert state_dim - lidar_dim == COM_feature_dim * COM_history_time_step, "Check COM_encoder output and state_encoder input in the cfg.yaml"

command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

COM_buffer = Buffer(env.num_envs, COM_history_time_step, COM_feature_dim)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['command_tracking']['architecture'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))

print("Loaded command tracking policy weight from {}\n".format(command_tracking_weight_path))

start = time.time()

# Load learned environment model weight
loaded_environment_model = Lidar_environment_model(COM_encoding_config=cfg["environment_model"]["architecture"]["COM_encoder"],
                                                   state_encoding_config=cfg["environment_model"]["architecture"]["state_encoder"],
                                                   command_encoding_config=cfg["environment_model"]["architecture"]["command_encoder"],
                                                   recurrence_config=cfg["environment_model"]["architecture"]["recurrence"],
                                                   prediction_config=cfg["environment_model"]["architecture"]["traj_predictor"],
                                                   device=device)
loaded_environment_model.load_state_dict(torch.load(weight_path, map_location=device)['model_architecture_state_dict'])
loaded_environment_model.eval()
loaded_environment_model.to(device)

print("Loaded environment model weight from {}\n".format(weight_path))

# Load sampler
n_prediction_step = int(cfg["CVAE"]["planner"]["prediction_period"] / cfg['command_tracking']['command_period'])
wo_cvae_sampler = Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal(command_range=cfg["environment"]["command"],
                                                                                  n_sample=cfg["CVAE"]["planner"]["wo_CVAE_number_of_sample"],
                                                                                  n_horizon=n_prediction_step,
                                                                                  n_bin=cfg["CVAE"]["planner"]["number_of_bin"],
                                                                                  beta=cfg["CVAE"]["planner"]["wo_CVAE_beta"],
                                                                                  gamma=cfg["CVAE"]["planner"]["gamma"],
                                                                                  sigma=cfg["CVAE"]["planner"]["sigma"],
                                                                                  noise_sigma=0.1,
                                                                                  noise=False,
                                                                                  action_dim=user_command_dim,
                                                                                  random_command_sampler=user_command)

w_cvae_sampler = CVAE_implicit_distribution_inference(state_encoding_config=cfg["CVAE"]["architecture"]["state_encoder"],
                                                      latent_decoding_config=cfg["CVAE"]["architecture"]["latent_decoder"],
                                                      recurrence_decoding_config=cfg["CVAE"]["architecture"]["recurrence_decoder"],
                                                      command_decoding_config=cfg["CVAE"]["architecture"]["command_decoder"],
                                                      device=device,
                                                      trained_weight=cvae_weight_path,
                                                      cfg_command=cfg["environment"]["command"])
w_cvae_sampler.eval()
w_cvae_sampler.to(device)
print("Loaded CVAE sampler weight from {}\n".format(cvae_weight_path))

# Set action planner
action_planner = Stochastic_action_planner_w_CVAE(wo_cvae_sampler=wo_cvae_sampler,
                                                  w_cvae_sampler=w_cvae_sampler,
                                                  wo_cvae_n_sample=cfg["CVAE"]["planner"]["wo_CVAE_number_of_sample"],
                                                  w_cvae_n_sample=cfg["CVAE"]["planner"]["CVAE_number_of_sample"],
                                                  n_prediction_step=n_prediction_step,
                                                  gamma=cfg["CVAE"]["planner"]["gamma"],
                                                  beta=cfg["CVAE"]["planner"]["CVAE_beta"])

# MUST safe period from collision
MUST_safety_period = 3.0
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['command_tracking']['command_period'])

# Set constant
collision_threshold = 0.05
goal_distance_threshold = 10
num_goals = cfg["environment"]["n_goals_per_env"]
if validation:
    init_seed = cfg["environment"]["seed"]["validate"]
    print("Validating ...")
else:
    init_seed = cfg["environment"]["seed"]["evaluate"]
    print("Evaluating ...")
goal_time_limit = 180.

# Make directory to save results
result_save_directory = f"{task_name}/Result/CVAE"
check_saving_folder(result_save_directory)

# Backup files
items_to_save = ["/cfg.yaml", "/CVAE_pgn.py"]
for item_to_save in items_to_save:
    save_location = task_path + "/../../../../" + result_save_directory + item_to_save
    copyfile(task_path + item_to_save, save_location)

# Set wandb logger
if cfg["logging"]:
    wandb.init(name="CVAE_"+task_name, project="Quadruped_RL")

pdb.set_trace()

print("<<-- Evaluating CVAE -->>")

for grid_size in [2.5, 3., 4.]:
    eval_start = time.time()

    # Set obstacle grid size
    cfg["environment"]["test_obstacle_grid_size"] = grid_size

    # Set empty list to log result
    n_total_case = cfg["environment"]["n_evaluate_envs"] * num_goals
    n_total_success_case = 0
    list_traversal_time = []
    list_traversal_distance = []
    list_num_collision = []
    list_success = []

    for env_id in range(cfg["environment"]["n_evaluate_envs"]):
        # Generate new environment with different seed (reset is automatically called)
        cfg["environment"]["seed"]["evaluate"] = env_id * 10 + init_seed
        env = VecEnv(simul_test.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)
        env.load_scaling(command_tracking_weight_dir, int(iteration_number))

        # Reset
        env.initialize_n_step()
        action_planner.reset()
        goal_position = env.set_goal()[np.newaxis, :]
        COM_buffer.reset()
        env.turn_on_visualization()

        # Initialize
        step = 0
        n_test_case = 0
        n_success_test_case = 0
        sample_user_command = np.zeros(3)
        goal_current_duration = 0.
        command_log = []
        collision_idx = []
        traversal_distance = 0.
        previous_coordinate = None
        current_coordinate = None

        while n_test_case < num_goals:
            frame_start = time.time()
            control_start = time.time()

            new_action_time = step % command_period_steps == 0

            # log traversal distance (just env 0)
            previous_coordinate = current_coordinate
            current_coordinate = env.coordinate_observe()
            if previous_coordinate is not None:
                delta_coordinate = current_coordinate[0, :-1] - previous_coordinate[0, :-1]
                traversal_distance += (delta_coordinate[0] ** 2 + delta_coordinate[1] ** 2) ** 0.5

            # observation before taking step
            obs, _ = env.observe(False)

            # update COM feature
            if step % COM_history_update_period == 0:
                COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                COM_buffer.update(COM_feature)

            if new_action_time:
                # prepare state
                init_coordinate_obs = env.coordinate_observe()
                lidar_data = obs[0, proprioceptive_sensor_dim:]
                goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
                current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
                goal_position_L *= np.clip(goal_distance_threshold / current_goal_distance, a_min=None, a_max=1.)
                goal_position_L = goal_position_L.astype(np.float32)
                current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
                COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
                state = np.concatenate((lidar_data, COM_history_feature)).astype(np.float32)

                # sample command sequences
                action_candidates = action_planner.sample(torch.from_numpy(state).unsqueeze(0).to(device), torch.from_numpy(goal_position_L).to(device))

                # simulate sampled command sequences
                state = np.tile(state, (cfg["CVAE"]["planner"]["wo_CVAE_number_of_sample"] + cfg["CVAE"]["planner"]["CVAE_number_of_sample"], 1))
                predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                   torch.from_numpy(action_candidates).to(device),
                                                                                   training=False)
                predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)

                delta_goal_distance = current_goal_distance - np.sqrt(np.sum(np.power(predicted_coordinates - goal_position_L, 2), axis=-1))

                goal_reward = np.sum(delta_goal_distance, axis=0)
                goal_reward -= np.min(goal_reward)
                goal_reward /= (np.max(goal_reward) + 1e-5)  # normalize reward

                safety_reward = 1 - predicted_P_cols
                safety_reward = np.mean(safety_reward, axis=0)
                safety_reward /= (np.max(safety_reward) + 1e-5)  # normalize reward

                reward = 1.0 * goal_reward * safety_reward + 0.3 * safety_reward
                # reward = 1.0 * goal_reward * safety_reward

                # exclude trajectory that collides with obstacle
                # (do not filter if all trajectories are predicted to collide because we need minimal reward for optimization)
                coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]
                if len(coll_idx) != (cfg["CVAE"]["planner"]["wo_CVAE_number_of_sample"] + cfg["CVAE"]["planner"]["CVAE_number_of_sample"]):
                    reward[coll_idx] = 0

                # receive final two candidate trajectories, each from different distribution
                wo_cvae_optimized_command_traj, w_cvae_optimized_command_traj = action_planner.seperate_update(reward)

                # simulate final two candidate trajectories
                final_candidate_traj = np.stack((wo_cvae_optimized_command_traj, w_cvae_optimized_command_traj), axis=1)
                predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state[:2, :]).to(device),
                                                                                   torch.from_numpy(final_candidate_traj).to(device),
                                                                                   training=False)
                # compute rewards of final two candidate trajectories
                predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                delta_goal_distance = current_goal_distance - np.sqrt(np.sum(np.power(predicted_coordinates - goal_position_L, 2), axis=-1))
                goal_reward = np.sum(delta_goal_distance, axis=0)
                goal_reward -= np.min(goal_reward)
                goal_reward /= (np.max(goal_reward) + 1e-5)  # normalize reward
                safety_reward = 1 - predicted_P_cols
                safety_reward = np.mean(safety_reward, axis=0)
                safety_reward /= (np.max(safety_reward) + 1e-5)  # normalize reward
                reward = 1.0 * goal_reward * safety_reward + 0.3 * safety_reward

                # exclude trajectory that collides with obstacle
                coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]
                if len(coll_idx) != 2:
                    reward[coll_idx] = 0

                # select higher reward trajectory
                if reward[0] >= reward[1]:
                    sample_user_command = wo_cvae_optimized_command_traj[0]
                    sample_user_command_traj = wo_cvae_optimized_command_traj
                else:
                    sample_user_command = wo_cvae_optimized_command_traj[0]
                    sample_user_command_traj = w_cvae_optimized_command_traj
                action_planner.set_optimized_result(sample_user_command_traj)

                # # plot predicted trajectory
                # traj_len, n_sample, coor_dim = predicted_coordinates.shape
                # for j in range(n_sample):
                #     plt.plot(predicted_coordinates[:, j, 0], predicted_coordinates[:, j, 1])
                # plt.savefig("sampled_traj (ours).png")
                # plt.clf()
                # pdb.set_trace()

                # simulate optimized command sequence
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

            # Execute first command in optimized command sequence using command tracking controller
            tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
            tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
            tracking_obs = tracking_obs.astype(np.float32)

            with torch.no_grad():
                tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
            
            control_end = time.time()

            _, done = env.step(tracking_action.cpu().detach().numpy())

            # Check collision
            collision = env.single_env_collision_check()
            if collision:
                collision_idx.append(step)

            # Update progress
            command_log.append(sample_user_command)
            step += 1
            goal_current_duration += cfg['environment']['control_dt']

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

            # if new_action_time:
            #    print(control_end - control_start)

            if cfg["realistic"]:
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)

            if goal_current_duration > goal_time_limit:
                done[0] = True

            # fail
            if done[0] == True:
                # Reset
                env.initialize_n_step()   # keep start in different initial condiition
                env.reset()
                action_planner.reset()
                goal_position = env.set_goal()[np.newaxis, :]
                COM_buffer.reset()

                # Update
                n_test_case += 1
                num_collision = compute_num_collision(collision_idx)
                # print(f"Intermediate result : {n_success_test_case} / {n_test_case} || Collision: {num_collision}")

                # Save result
                list_num_collision.append(num_collision)
                list_success.append(False)

                # Initialize
                step = 0
                sample_user_command = np.zeros(3)
                goal_current_duration = 0.
                command_log = []
                collision_idx = []
                traversal_distance = 0.
                previous_coordinate = None
                current_coordinate = None
            # success
            elif current_goal_distance < 0.5:
                # Reset
                env.initialize_n_step()  # keep start in different initial condiition
                env.reset()
                action_planner.reset()
                goal_position = env.set_goal()[np.newaxis, :]
                COM_buffer.reset()

                # Update
                n_test_case += 1
                n_success_test_case += 1
                num_collision = compute_num_collision(collision_idx)
                # print(f"Intermediate result : {n_success_test_case} / {n_test_case} || Collision: {num_collision} || Number of steps: {step} || Traversal distance: {traversal_distance}")

                # Save result
                list_traversal_time.append(step * cfg['environment']['control_dt'])
                list_traversal_distance.append(traversal_distance)
                list_num_collision.append(num_collision)
                list_success.append(True)

                # Plot command
                if cfg["plot_command"]:
                    plot_command_result(command_traj=np.array(command_log),
                                        folder_name="command_trajectory",
                                        task_name=task_name,
                                        run_name=f"CVAE_{str(grid_size)}",
                                        n_update=n_test_case + num_goals * env_id,
                                        control_dt=cfg["environment"]["control_dt"])

                # Initialize
                step = 0
                sample_user_command = np.zeros(3)
                goal_current_duration = 0.
                command_log = []
                collision_idx = []
                traversal_distance = 0.
                current_coordinate = None
                previous_coordinate = None

        n_total_success_case += n_success_test_case

    assert len(list_traversal_time) == n_total_success_case
    assert len(list_traversal_distance) == n_total_success_case
    assert len(list_num_collision) == n_total_case
    assert len(list_success) == n_total_case

    success_rate = n_total_success_case / n_total_case
    list_traversal_time = np.array(list_traversal_time)
    list_traversal_distance = np.array(list_traversal_distance)
    list_num_collision = np.array(list_num_collision)
    list_success = np.array(list_success)

    # Compute statistical indicators
    quantile_percent = [25, 50, 75]
    traversal_time_quantile = np.percentile(list_traversal_time, quantile_percent)
    traversal_time_mean = np.mean(list_traversal_time)
    traversal_time_std = np.std(list_traversal_time)
    traversal_distance_quantile = np.percentile(list_traversal_distance, quantile_percent)
    traversal_distance_mean = np.mean(list_traversal_distance)
    traversal_distance_std = np.std(list_traversal_distance)
    num_collision_quantile = np.percentile(list_num_collision, quantile_percent)
    num_collision_mean = np.mean(list_num_collision)
    num_collision_std = np.std(list_num_collision)

    # Save summarized result
    final_result = defaultdict(dict)
    final_result["SR"]["ratio"] = success_rate
    final_result["SR"]["n_success"] = n_total_success_case
    final_result["SR"]["n_total"] = n_total_case
    final_result["Time"]["mean"] = traversal_time_mean
    final_result["Time"]["std"] = traversal_time_std
    final_result["Time"]["q1"] = traversal_time_quantile[0]
    final_result["Time"]["q2"] = traversal_time_quantile[1]
    final_result["Time"]["q3"] = traversal_time_quantile[2]
    final_result["Distance"]["mean"] = traversal_distance_mean
    final_result["Distance"]["std"] = traversal_distance_std
    final_result["Distance"]["q1"] = traversal_distance_quantile[0]
    final_result["Distance"]["q2"] = traversal_distance_quantile[1]
    final_result["Distance"]["q3"] = traversal_distance_quantile[2]
    final_result["Num_collision"]["mean"] = num_collision_mean
    final_result["Num_collision"]["std"] = num_collision_std
    final_result["Num_collision"]["q1"] = num_collision_quantile[0]
    final_result["Num_collision"]["q2"] = num_collision_quantile[1]
    final_result["Num_collision"]["q3"] = num_collision_quantile[2]
    with open(f"{result_save_directory}/{str(grid_size)}_grid_result.json", "w") as f:
        json.dump(final_result, f)

    wandb.log(final_result)

    # Save raw result
    np.savez_compressed(f"{result_save_directory}/{str(grid_size)}_grid_result", time=list_traversal_time,
                        distance=list_traversal_distance, num_collision=list_num_collision,
                        success=list_success)

    eval_end = time.time()

    elapse_time_seconds = eval_end - eval_start
    elaspe_time_minutes = int(elapse_time_seconds / 60)
    elapse_time_seconds -= (elaspe_time_minutes * 60)
    elapse_time_seconds = int(elapse_time_seconds)

    print("===========================================")
    print(f"Grid_{str(grid_size)}:")
    print(f"SR: {n_total_success_case} / {n_total_case}")
    print(f"Time: {round(traversal_time_mean, 1)}  [{round(traversal_time_quantile[0], 1)} / {round(traversal_time_quantile[1], 1)} / {round(traversal_time_quantile[2], 1)}]")
    print(f"Distance: {round(traversal_distance_mean, 1)}  [{round(traversal_distance_quantile[0], 1)} / {round(traversal_distance_quantile[1], 1)} / {round(traversal_distance_quantile[2], 1)}]")
    print(f"Num_collision: {round(num_collision_mean, 1)}  [{round(num_collision_quantile[0], 1)} / {round(num_collision_quantile[1], 1)} / {round(num_collision_quantile[2], 1)}]")
    print(f"Elapsed time: {elaspe_time_minutes}m {elapse_time_seconds}s")

# env.stop_video_recording()
env.turn_off_visualization()
