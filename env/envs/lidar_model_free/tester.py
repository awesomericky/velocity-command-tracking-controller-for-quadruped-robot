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
import torch.nn as nn
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
from raisimGymTorch.env.envs.lidar_model.model import MLP
import argparse
import numpy as np
import datetime
import pdb
from collections import Counter
import random

def transform_coordinate_WL(w_init_coordinate, w_coordinate_traj):
    """
    Transform WORLD frame coordinate trajectory to LOCAL frame coordinate trajectory
    (WORLD frame --> LOCAL frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim) or (n_env, coordinate_dim)
    :param w_coordinate_traj: coordintate trajectory in WORLD frame (n_step, coordinate_dim) or (n_env, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    l_coordinate_traj = w_coordinate_traj - w_init_coordinate[:, :-1]
    l_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix.T)
    return l_coordinate_traj

"""
Point goal navigation test in single environment (just for visualization)

** In point_goal_initialize, hm_sizeX and hm_sizeY should be bigger than 30 because goal distance bigger than 10, 10, 5 are stored. Unless error will occur.
"""

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# task specification
task_name = "lidar_model_free"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, required=True)
parser.add_argument('-tw', '--tracking_weight', help='pre-trained command tracking policy weight path', type=str, required=True)
parser.add_argument('-pw', '--pretrained_latent_weight', help='pre-trained latent state weight path', type=str, default='')
args = parser.parse_args()
weight_path = args.weight
command_tracking_weight_path = args.tracking_weight
latent_state_weight = args.pretrained_latent_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

assert cfg["environment"]["determine_env"] != 0, "Environment should be determined to a single type"
assert cfg["environment"]["evaluate"], "Change cfg[environment][evaluate] to True"
assert not cfg["environment"]["random_initialize"], "Change cfg[environment][random_initialize] to True"
assert cfg["environment"]["point_goal_initialize"], "Change cfg[environment][point_goal_initialize] to True"

use_latent_state = cfg["architecture"]["use_latent_state"]

if use_latent_state:
    # Load pretrained latent state weight
    assert latent_state_weight != '', "Latent state weight not provided."
    pretrained_weight = torch.load(latent_state_weight, map_location=device)["model_architecture_state_dict"]
    latent_state_dict = dict()
    for k, v in pretrained_weight.items():
        if k.split('.', 1)[0] == "state_encoder":
            latent_state_dict[k.split('.', 1)[1]] = v
    assert len(latent_state_dict.keys()) != 0, "Error when loading weights"

    state_encoder_config = cfg["architecture"]["state_encoder"]
    activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
    state_encoder = MLP(state_encoder_config["shape"],
                        activation_map[state_encoder_config["activation"]],
                        state_encoder_config["input"],
                        state_encoder_config["output"],
                        dropout=state_encoder_config["dropout"],
                        batchnorm=state_encoder_config["batchnorm"])
    state_encoder_state_dict = state_encoder.state_dict()
    state_encoder_state_dict.update(latent_state_dict)
    state_encoder.load_state_dict(state_encoder_state_dict)
    state_encoder.eval()
    state_encoder.to(device)

env = VecEnv(lidar_model_free.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Use naive concatenation for encoding COM vel history
COM_feature_dim = 9
COM_history_time_step = 10
COM_history_update_period = int(0.05 / cfg["environment"]["control_dt"])
goal_pos_dim = 2

if use_latent_state:
    planning_ob_dim = cfg["architecture"]["state_encoder"]["output"] + goal_pos_dim
    assert cfg["architecture"]["state_encoder"]["input"] == lidar_dim + COM_feature_dim * COM_history_time_step, "State encoder input dimension does not match with obsevation dimension"
else:
    planning_ob_dim = lidar_dim + COM_feature_dim * COM_history_time_step + goal_pos_dim
planning_act_dim = user_command_dim
command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['environment']['command_period'] / cfg['environment']['control_dt'])
num_envs = cfg['environment']['num_envs']
assert n_steps % command_period_steps == 0, "Total steps in training should be divided by command period steps."
assert n_steps % COM_history_update_period == 0, "Total steps in training should be divided by COM history update period steps"

COM_buffer = Buffer(num_envs, COM_history_time_step, COM_feature_dim)

action_clipping_range = np.array([[cfg["environment"]["command"]["forward_vel"]["min"], cfg["environment"]["command"]["forward_vel"]["max"]],
                                  [cfg["environment"]["command"]["lateral_vel"]["min"], cfg["environment"]["command"]["lateral_vel"]["max"]],
                                  [cfg["environment"]["command"]["yaw_rate"]["min"], cfg["environment"]["command"]["yaw_rate"]["max"]]])

# Load trained planning policy weight
loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, planning_ob_dim, planning_act_dim)
loaded_graph.load_state_dict(torch.load(weight_path, map_location=device)['actor_architecture_state_dict'])
loaded_graph.eval()
loaded_graph.to(device)
planning_weight_dir = weight_path.rsplit('/', 1)[0] + '/'
planning_iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
command_tracking_iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]

# Set and load runnning mean and variance
env.set_running_mean_var(first_type_dim=[num_envs, planning_ob_dim],
                         second_type_dim=[num_envs, command_tracking_ob_dim])
env.load_scaling(planning_weight_dir, int(planning_iteration_number), type=1)
env.load_scaling(command_tracking_weight_dir, int(command_tracking_iteration_number), type=2)

goal_distance_threshold = 10.
final_activation = nn.Tanh()

total_n_goals = 12

print("Loaded weight from {}\n".format(weight_path))
start = time.time()
n_success = 0

pdb.set_trace()

for goal_id in range(total_n_goals):
    env.initialize_n_step()
    goal_position = env.set_goal()[np.newaxis, :]
    env.reset()
    COM_buffer.reset()

    reward_sum = 0
    step = 0

    while True:
        frame_start = time.time()

        new_command_time = step % command_period_steps == 0

        if new_command_time:
            # save coordinate before taking step to modify the labeled data
            coordinate_obs = env.coordinate_observe()

        obs, _ = env.observe(False)  # observation before taking step
        if step % COM_history_update_period == 0:
            COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
            COM_buffer.update(COM_feature)

        if new_command_time:
            lidar_data = obs[:, proprioceptive_sensor_dim:]
            COM_history = COM_buffer.return_data(flatten=True)

            # prepare goal position
            goal_position_L = transform_coordinate_WL(coordinate_obs, goal_position)
            current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2), axis=-1))[:, np.newaxis]
            goal_position_L *= np.clip(goal_distance_threshold / current_goal_distance, a_min=None, a_max=1.)

            if use_latent_state:
                temp_state = np.concatenate((lidar_data, COM_history), axis=1).astype(np.float32)
                planning_obs = state_encoder.architecture(torch.from_numpy(temp_state).to(device))
                planning_obs = planning_obs.cpu().detach().numpy()
                planning_obs = np.concatenate((planning_obs, goal_position_L), axis=1)
            else:
                planning_obs = np.concatenate((lidar_data, COM_history, goal_position_L), axis=1)

            planning_obs = env.force_normalize_observation(planning_obs, type=1)
            planning_obs = planning_obs.astype(np.float32)
            sample_user_command = loaded_graph.architecture(torch.from_numpy(planning_obs).to(device))
            sample_user_command = final_activation(sample_user_command).cpu().detach().numpy() * action_clipping_range[:, 1]

        tracking_obs = np.concatenate((sample_user_command, obs[:, :proprioceptive_sensor_dim]), axis=1)
        tracking_obs = env.force_normalize_observation(tracking_obs, type=2).astype(np.float32)
        with torch.no_grad():
            tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
        rewards, dones = env.partial_step(tracking_action.cpu().detach().numpy())

        coordinate_obs = env.coordinate_observe()  # coordinate after taking step

        # sum reward
        reward_sum += rewards[0]

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

        step += 1

        if dones[0]:
            break

        if current_goal_distance < 0.5:
            n_success += 1
            print(f"{goal_id}/{total_n_goals}: Reward={reward_sum} || Step={step}")
            break

end = time.time()

print(f"Success rate: {n_success}/{total_n_goals}")