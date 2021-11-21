import matplotlib.pyplot as plt
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_enviroment_model_param, UserCommand
from raisimGymTorch.helper.utils_plot import plot_trajectory_prediction_result
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
import wandb
from raisimGymTorch.env.envs.lidar_model.model import Lidar_environment_model
from raisimGymTorch.env.envs.lidar_model.trainer import Trainer, Trainer_TCN
from raisimGymTorch.env.envs.lidar_model.action import Command_sampler, Time_correlated_command_sampler, Normal_time_correlated_command_sampler
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
from raisimGymTorch.env.envs.lidar_model.model import CVAE_implicit_distribution_inference


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

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim) or (n_env, coordinate_dim)
    :param w_coordinate_traj: coordintate trajectory in WORLD frame (n_step, coordinate_dim) or (n_env, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    l_coordinate_traj = w_coordinate_traj - w_init_coordinate[:, :-1]
    l_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix.T)
    return l_coordinate_traj

np.random.seed(1)

# task specification
task_name = "CVAE_lidar_environment_model_evaluate"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='trained environment model weight path', type=str, required=True)
parser.add_argument('-cw', '--cvae_weight', help='trained CVAE model weight path', type=str, required=True)
parser.add_argument('-tw', '--tracking_weight', help='trained command tracking policy weight path', type=str, required=True)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight
cvae_weight_path = args.cvae_weight
command_tracking_weight_path = args.tracking_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config (load saved configuration)
# cfg = YAML().load(open(weight_path.rsplit("/", 1)[0] + "/cfg.yaml", 'r'))
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

assert cfg["environment"]["determine_env"] == 0, "Environment should not be determined to a single type"
assert cfg["environment"]["evaluate"], "Change cfg[environment][evaluate] to True"
assert not cfg["environment"]["random_initialize"], "Change cfg[environment][random_initialize] to False"
assert not cfg["environment"]["point_goal_initialize"], "Change cfg[environment][point_goal_initialize] to False"
assert not cfg["environment"]["CVAE_data_collection_initialize"], "Change cfg[environment][ CVAE_data_collection_initialize] to False"
assert not cfg["environment"]["safe_control_initialize"], "Change cfg[environment][safe_control_initialize] to False"
assert cfg["environment"]["CVAE_environment_initialize"], "Change cfg[environment][CVAE_environment_evaluation_initialize] to True"

cfg['environment']['num_threads'] = cfg['environment']['evaluate_num_threads']

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

environment_model = Lidar_environment_model(COM_encoding_config=cfg["architecture"]["COM_encoder"],
                                            state_encoding_config=cfg["architecture"]["state_encoder"],
                                            command_encoding_config=cfg["architecture"]["command_encoder"],
                                            recurrence_config=cfg["architecture"]["recurrence"],
                                            prediction_config=cfg["architecture"]["traj_predictor"],
                                            device=device)

# Generate trainer for evaluation
trainer = Trainer(environment_model=environment_model,
                  state_dim=state_dim,
                  command_dim=command_dim,
                  P_col_dim=P_col_dim,
                  coordinate_dim=coordinate_dim,
                  prediction_period=cfg["data_collection"]["prediction_period"],
                  delta_prediction_time=cfg["data_collection"]["command_period"],
                  loss_weight=cfg["training"]["loss_weight"],
                  max_storage_size=cfg["training"]["storage_size"],
                  num_learning_epochs=cfg["training"]["num_epochs"],
                  mini_batch_size=cfg["training"]["batch_size"],
                  shuffle_batch=cfg["training"]["shuffle_batch"],
                  clip_grad=cfg["training"]["clip_gradient"],
                  learning_rate=cfg["training"]["learning_rate"],
                  max_grad_norm=cfg["training"]["max_gradient_norm"],
                  device=device,
                  logging=False,
                  P_col_interpolate=cfg["training"]["interpolate_probability"],
                  prioritized_data_update=cfg["data_collection"]["prioritized_data_update"],
                  prioritized_data_update_magnitude=cfg["data_collection"]["prioritized_data_update_magnitude"])

# Load trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))

# Load trained environment model
loaded_environment_model = Lidar_environment_model(COM_encoding_config=cfg["architecture"]["COM_encoder"],
                                                   state_encoding_config=cfg["architecture"]["state_encoder"],
                                                   command_encoding_config=cfg["architecture"]["command_encoder"],
                                                   recurrence_config=cfg["architecture"]["recurrence"],
                                                   prediction_config=cfg["architecture"]["traj_predictor"],
                                                   device=device)
loaded_environment_model.load_state_dict(torch.load(weight_path, map_location=device)['model_architecture_state_dict'])
loaded_environment_model.eval()
loaded_environment_model.to(device)

# Load CVAE inference model (Learned command sampling distribution)
w_cvae_sampler = CVAE_implicit_distribution_inference(state_encoding_config=cfg["CVAE_architecture"]["state_encoder"],
                                                      latent_decoding_config=cfg["CVAE_architecture"]["latent_decoder"],
                                                      recurrence_decoding_config=cfg["CVAE_architecture"]["recurrence_decoder"],
                                                      command_decoding_config=cfg["CVAE_architecture"]["command_decoder"],
                                                      device=device,
                                                      trained_weight=cvae_weight_path,
                                                      cfg_command=cfg["environment"]["command"])
w_cvae_sampler.eval()
w_cvae_sampler.to(device)

final_P_col_accuracy = []
final_col_accuracy = []
final_not_col_accuracy = []
final_coordinate_error = []
num_test = 100

goal_distance_threshold = 10.
n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])

# env.turn_off_visualization()

pdb.set_trace()

for n_test in range(num_test):
    env.initialize_n_step()
    goal_position = env.parallel_set_goal()
    env.reset()
    COM_buffer.reset()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

    COM_history_traj = []
    lidar_traj = []
    state_traj = []
    command_traj = []
    P_col_traj = []
    coordinate_traj = []
    init_coordinate_traj = []
    done_envs = set()

    # sample_user_command = user_command.uniform_sample_evaluate()

    for step in range(n_steps):
        frame_start = time.time()
        new_command_time = step % command_period_steps == 0
        traj_update_time = (step + 1) % command_period_steps == 0

        if new_command_time:
            env.initialize_n_step()  # to reset in new position
            env.partial_reset(list(done_envs))  # reset only terminated environment

            # save coordinate before taking step to modify the labeled data
            coordinate_obs = env.coordinate_observe()
            init_coordinate_traj.append(coordinate_obs)

        obs, _ = env.observe(False)  # observation before taking step
        if step % COM_history_update_period == 0:
            COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
            COM_buffer.update(COM_feature)

        if new_command_time:
            done_envs = set()
            previous_done_envs = np.array([])
            temp_state = np.zeros((cfg['environment']['num_envs'], state_dim))
            temp_lidar = np.zeros((cfg['environment']['num_envs'], lidar_dim))
            temp_command = np.zeros((cfg['environment']['num_envs'], command_dim))
            temp_P_col = np.zeros(cfg['environment']['num_envs'])
            temp_coordinate = np.zeros((cfg['environment']['num_envs'], coordinate_dim))

            # prepare state
            lidar_data = obs[:, proprioceptive_sensor_dim:]
            temp_COM_history = COM_buffer.return_data(flatten=True)
            temp_state = np.concatenate((lidar_data, temp_COM_history), axis=1)

            # prepare goal position
            goal_position_L = transform_coordinate_WL(coordinate_obs, goal_position)
            current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2), axis=-1))[:, np.newaxis]
            goal_position_L *= np.clip(goal_distance_threshold / current_goal_distance, a_min=None, a_max=1.)

            # sample command
            sampled_command_traj = w_cvae_sampler(torch.from_numpy(temp_state.astype(np.float32)).to(device),
                                                  torch.from_numpy(goal_position_L.astype(np.float32)).to(device),
                                                  1, n_prediction_step, return_torch=False)
            sample_user_command = sampled_command_traj[0, :, 0, :]
            temp_command = sample_user_command.copy()

        tracking_obs = np.concatenate((sample_user_command, obs[:, :proprioceptive_sensor_dim]), axis=1)
        tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
        with torch.no_grad():
            tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
        _, dones = env.partial_step(tracking_action.cpu().detach().numpy())

        coordinate_obs = env.coordinate_observe()  # coordinate after taking step

        # update P_col and coordinate for terminated environment
        current_done_envs = np.where(dones == 1)[0]
        counter_current_done_envs = Counter(current_done_envs)
        counter_previous_done_envs = Counter(previous_done_envs)
        new_done_envs = np.array(sorted((counter_current_done_envs - counter_previous_done_envs).elements())).astype(int)
        done_envs.update(new_done_envs)
        previous_done_envs = current_done_envs
        temp_P_col[new_done_envs] = dones[new_done_envs].astype(int)
        temp_coordinate[new_done_envs, :] = coordinate_obs[new_done_envs, :-1]

        # reset COM buffer for terminated environment
        COM_buffer.partial_reset(current_done_envs)

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

        if traj_update_time:
            # update P_col and coordinate for not terminated environment
            counter_current_done_envs = Counter(list(done_envs))
            counter_default_envs = Counter(np.arange(cfg['environment']['num_envs']))
            not_done_envs = np.array(sorted((counter_default_envs - counter_current_done_envs).elements())).astype(int)
            temp_P_col[not_done_envs] = 0
            temp_coordinate[not_done_envs, :] = coordinate_obs[not_done_envs, :-1]

            state_traj.append(temp_state)
            command_traj.append(temp_command)
            P_col_traj.append(temp_P_col)
            coordinate_traj.append(temp_coordinate)

    state_traj = np.array(state_traj)
    command_traj = np.array(command_traj)
    P_col_traj = np.array(P_col_traj)
    coordinate_traj = np.array(coordinate_traj)
    init_coordinate_traj = np.array(init_coordinate_traj)

    (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (mean_total_col_prediction_accuracy, mean_col_prediction_accuracy, mean_not_col_prediction_accuracy, mean_coordinate_error) \
        = trainer.evaluate(environment_model=loaded_environment_model,
                           state_traj=state_traj,
                           command_traj=command_traj,
                           dones_traj=P_col_traj,
                           coordinate_traj=coordinate_traj,
                           init_coordinate_traj=init_coordinate_traj,
                           collision_threshold=0.05)

    final_P_col_accuracy.append(mean_total_col_prediction_accuracy)
    if mean_col_prediction_accuracy != -1:
        final_col_accuracy.append(mean_col_prediction_accuracy)
    if mean_not_col_prediction_accuracy != -1:
        final_not_col_accuracy.append(mean_not_col_prediction_accuracy)
    final_coordinate_error.append(mean_coordinate_error)

    print('====================================================')
    print('{:>6}th evaluation'.format(n_test))
    print('{:<40} {:>6}'.format("total collision accuracy: ", '{:0.6f}'.format(mean_total_col_prediction_accuracy)))
    print('{:<40} {:>6}'.format("collision accuracy: ", '{:0.6f}'.format(mean_col_prediction_accuracy)))
    print('{:<40} {:>6}'.format("no collision accuracy: ", '{:0.6f}'.format(mean_not_col_prediction_accuracy)))
    print('{:<40} {:>6}'.format("coordinate error: ", '{:0.6f}'.format(mean_coordinate_error)))

    print('====================================================\n')

env.turn_off_visualization()

type = "after"
# type = "before"

final_P_col_accuracy = np.array(final_P_col_accuracy)
final_coordinate_error = np.array(final_coordinate_error)
np.savez_compressed(f"Lidar_2D_model_result_{type}", collision=final_P_col_accuracy, coordinate=final_coordinate_error)

plt.scatter(final_coordinate_error, final_P_col_accuracy, s=15)
plt.title("Lidar 2D environment prediction result")
plt.xlabel("Coordinate error [m]")
plt.ylabel("Collision accuracy")
plt.ylim(-0.1, 1.1)
plt.savefig(f"Lidar_2D_model_result_{type}")
plt.clf()

final_col_accuracy = np.array(final_col_accuracy)
final_not_col_accuracy = np.array(final_not_col_accuracy)

print("--------------------------------------------------")
print(f"Collision: {np.mean(final_P_col_accuracy)}")
print(f"Collision(O): {np.mean(final_col_accuracy)}")
print(f"Collision(X): {np.mean(final_not_col_accuracy)}")
print(f"Coodinate: {np.mean(final_coordinate_error)}")
print("--------------------------------------------------")
