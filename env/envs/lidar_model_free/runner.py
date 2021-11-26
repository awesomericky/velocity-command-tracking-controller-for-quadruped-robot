from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import lidar_model_free
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher, UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_tracking_result
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
from collections import defaultdict
import pdb
import wandb
from raisimGymTorch.env.envs.lidar_model.storage import Buffer
from raisimGymTorch.env.envs.lidar_model.model import MLP
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

def curriculum_update(current_scale, power_factor):
    return current_scale ** power_factor

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# task specification
task_name = "lidar_model_free"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('-tw', '--tracking_weight', help='pre-trained command tracking policy weight path', type=str, required=True)
parser.add_argument('-pw', '--pretrained_latent_weight', help='pre-trained latent state weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
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
reward_names = list(map(str, cfg['environment']['reward'].keys()))
reward_names.append('reward_sum')

assert cfg["environment"]["determine_env"] == 0, "Environment should not be determined to a single type"
assert not cfg["environment"]["evaluate"], "Change cfg[environment][evaluate] to False"
assert cfg["environment"]["random_initialize"], "Change cfg[environment][random_initialize] to True"
assert not cfg["environment"]["point_goal_initialize"], "Change cfg[environment][point_goal_initialize] to False"

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

# create environment from the configuration file
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
evaluate_n_steps = n_steps
command_period_steps = math.floor(cfg['environment']['command_period'] / cfg['environment']['control_dt'])
num_envs = cfg['environment']['num_envs']
assert n_steps % command_period_steps == 0, "Total steps in training should be divided by command period steps."
assert n_steps % COM_history_update_period == 0, "Total steps in training should be divided by COM history update period steps"

COM_buffer = Buffer(num_envs, COM_history_time_step, COM_feature_dim)

# Log the training and evaluating process or not
logging = cfg["logging"]

# wandb initialize
if logging:
    if mode == 'retrain':
        wandb.init(name=task_name + "_retrain", project="Quadruped_RL")
    else:
        wandb.init(name=task_name, project="Quadruped_RL")

action_clipping_range = np.array([[cfg["environment"]["command"]["forward_vel"]["min"], cfg["environment"]["command"]["forward_vel"]["max"]],
                                  [cfg["environment"]["command"]["lateral_vel"]["min"], cfg["environment"]["command"]["lateral_vel"]["max"]],
                                  [cfg["environment"]["command"]["yaw_rate"]["min"], cfg["environment"]["command"]["yaw_rate"]["max"]]])

actor = ppo_module.Actor_two_side_clip(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, planning_ob_dim, planning_act_dim),
                                       ppo_module.MultivariateGaussianDiagonalCovariance_two_side_clip(planning_act_dim, list(action_clipping_range[:, 1] * 0.2), action_clipping_range, device),
                                       action_clipping_range,
                                       device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, planning_ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=num_envs,
              num_transitions_per_env=n_steps,
              num_learning_epochs=cfg["training"]["num_learning_epochs"],
              num_mini_batches=cfg["training"]["num_mini_batches"],
              gamma=cfg["training"]["gamma"],
              lam=cfg["training"]["lamda"],
              entropy_coef=cfg["training"]["entropy_coef"],
              learning_rate=cfg["training"]["learning_rate"],
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=cfg["training"]["shuffle_batch"])

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]

# Set and load runnning mean and variance
env.set_running_mean_var(first_type_dim=[num_envs, planning_ob_dim],
                         second_type_dim=[num_envs, command_tracking_ob_dim])
env.load_scaling(command_tracking_weight_dir, int(iteration_number), type=2)

goal_distance_threshold = 10.
final_activation = nn.Tanh()

# curriculum learning
init_curriculum_scale = 0.1
curriculum_power_factor = 0.997
current_curriculum_scale = init_curriculum_scale

pdb.set_trace()

for update in range(cfg["environment"]["max_n_update"]):

    # evaluate
    if update % cfg['environment']['eval_every_n'] == 0:
        start = time.time()

        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, planning_ob_dim, planning_act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt', map_location=device)['actor_architecture_state_dict'])
        loaded_graph.eval()
        loaded_graph.to(device)

        env.initialize_n_step()
        goal_position = env.parallel_set_goal()
        env.reset()
        COM_buffer.reset()
        # env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        done_envs = set()
        done_envs_storage = set()
        total_n_plan_steps = int(evaluate_n_steps / command_period_steps)
        total_n_track_steps = evaluate_n_steps
        reward_sum = np.zeros(num_envs, dtype=np.float32)
        reward_trajectory = np.zeros((num_envs, total_n_plan_steps, cfg['environment']['n_rewards'] + 1))
        reward_w_coeff_trajectory = np.zeros((num_envs, total_n_plan_steps, cfg['environment']['n_rewards'] + 1))

        for step in range(evaluate_n_steps):
            frame_start = time.time()

            new_command_time = step % command_period_steps == 0
            update_time = (step + 1) % command_period_steps == 0

            if new_command_time:
                # reset only terminated environment
                env.initialize_n_step()  # to reset in new position
                env.partial_reset(list(done_envs))

                # save coordinate before taking step to modify the labeled data
                coordinate_obs = env.coordinate_observe()

            obs, _ = env.observe(False)  # observation before taking step
            if step % COM_history_update_period == 0:
                COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                COM_buffer.update(COM_feature)

            if new_command_time:
                done_envs = set()
                previous_done_envs = np.array([])

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

            # update P_col and coordinate for terminated environment
            current_done_envs = np.where(dones == 1)[0]
            counter_current_done_envs = Counter(current_done_envs)
            counter_previous_done_envs = Counter(previous_done_envs)
            new_done_envs = np.array(sorted((counter_current_done_envs - counter_previous_done_envs).elements())).astype(int)
            done_envs.update(new_done_envs)
            previous_done_envs = current_done_envs.copy()

            # sum reward
            reward_sum[new_done_envs] += rewards[new_done_envs]
            counter_total_envs = Counter(np.arange(num_envs))
            not_done_envs = np.array(sorted((counter_total_envs - counter_current_done_envs).elements())).astype(int)
            reward_sum[not_done_envs] += rewards[not_done_envs]

            env.reward_logging(cfg['environment']['n_rewards'] + 1)
            reward_trajectory[new_done_envs, int(step / command_period_steps), :] += env.reward_log[new_done_envs, :]
            reward_trajectory[not_done_envs, int(step / command_period_steps), :] += env.reward_log[not_done_envs, :]
            reward_w_coeff_trajectory[new_done_envs, int(step / command_period_steps), :] += env.reward_w_cpeff_log[new_done_envs, :]
            reward_w_coeff_trajectory[not_done_envs, int(step / command_period_steps), :] += env.reward_w_cpeff_log[not_done_envs, :]

            # sum done
            if update_time:
                done_envs_storage.update(np.array(list(done_envs)))

            # reset COM buffer for terminated environment
            COM_buffer.partial_reset(current_done_envs)

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

            if wait_time > 0.:
                time.sleep(wait_time)

        end = time.time()

        # env.turn_off_visualization()
        env.save_scaling(saver.data_dir, str(update), type=1)

        mean_reward_sum = np.mean(reward_sum)
        mean_done_sum = len(list(done_envs_storage)) / num_envs

        # reward logging (value & std)
        reward_trajectory_mean = np.mean(reward_trajectory, axis=0)  # (n_plan_steps, cfg['environment']['n_rewards'] + 1)
        reward_w_coeff_trajectory_mean = np.mean(reward_w_coeff_trajectory, axis=0)  # (n_plan_steps, cfg['environment']['n_rewards'] + 1)
        reward_trajectory_mean[:, -1] = reward_w_coeff_trajectory_mean[:, -1]  # log reward sum considering the reward coeff
        reward_mean = np.mean(reward_trajectory_mean, axis=0)  # average reward for each planning step
        assert reward_mean.shape[0] == cfg['environment']['n_rewards'] + 1

        if logging:
            logging_data = dict()
            logging_data["Evaluate/Mean_reward"] = mean_reward_sum
            for name, value in zip(reward_names, reward_mean):
                logging_name = f"Evaluate/Reward/value/{name}"
                logging_data[logging_name] = value
            logging_data["Evaluate/Mean_done"] = mean_done_sum
            wandb.log(logging_data)

        print('====================================================')
        print('{:>6}th evaluation'.format(update))
        print('{:<40} {:>6}'.format("average reward sum: ", '{:0.10f}'.format(mean_reward_sum)))
        print('{:<40} {:>6}'.format("average dones: ", '{:0.6f}'.format(mean_done_sum)))
        print('{:<40} {:>6}'.format("elapsed time: ", '{:6.4f}'.format(end - start)))
        print('====================================================\n')


    # generate new environment
    if update % cfg["environment"]["new_environment_every_n"] == 0:
        print("Sample new environment")
        # save previous running mean and std count
        obs_mean, obs_var, obs_count = env.get_running_mean_var_explicit(type=1)

        # curriculum learning
        current_curriculum_scale = curriculum_update(current_scale=current_curriculum_scale, power_factor=curriculum_power_factor)
        cfg["environment"]["obstacle_grid_size"]["min"] = cfg["environment"]["obstacle_grid_size"]["max"] - 2.5 * current_curriculum_scale  # 4.75 ~ 5.0 ==> 2.5 ~ 5.0
        cfg["environment"]["corridor_short_width"]["min"] = cfg["environment"]["corridor_short_width"]["max"] - 4.0 * current_curriculum_scale  # 5.6 ~ 6.0 ==? 2.0 ~ 6.0

        # create environment from the configuration file
        cfg["environment"]["seed"]["train"] = update + 2000
        env = VecEnv(lidar_model_free.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)
        # Set and load runnning mean and variance
        env.set_running_mean_var(first_type_dim=[num_envs, planning_ob_dim],
                                 second_type_dim=[num_envs, command_tracking_ob_dim])
        env.set_running_mean_var_explicit(obs_mean, obs_var, obs_count, type=1)
        env.load_scaling(command_tracking_weight_dir, int(iteration_number), type=2)

    start = time.time()

    # actual training
    env.initialize_n_step()
    goal_position = env.parallel_set_goal()
    env.reset()
    COM_buffer.reset()

    done_envs = set()
    done_envs_storage = set()
    total_n_plan_steps = int(n_steps / command_period_steps)
    total_n_track_steps = n_steps
    reward_sum = np.zeros(num_envs, dtype=np.float32)
    reward_trajectory = np.zeros((num_envs, total_n_plan_steps, cfg['environment']['n_rewards'] + 1))
    reward_w_coeff_trajectory = np.zeros((num_envs, total_n_plan_steps, cfg['environment']['n_rewards'] + 1))

    for step in range(n_steps):
        new_command_time = step % command_period_steps == 0
        update_time = (step + 1) % command_period_steps == 0

        if new_command_time:
            # reset only terminated environment
            env.initialize_n_step()  # to reset in new position
            env.partial_reset(list(done_envs))

            # save coordinate before taking step to modify the labeled data
            coordinate_obs = env.coordinate_observe()

        obs, _ = env.observe(False)  # observation before taking step
        if step % COM_history_update_period == 0:
            COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
            COM_buffer.update(COM_feature)

        if new_command_time:
            done_envs = set()
            previous_done_envs = np.array([])

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

            env.force_update_ob_rms(planning_obs, type=1)
            planning_obs = env.force_normalize_observation(planning_obs, type=1).astype(np.float32)
            sample_user_command = ppo.observe(planning_obs)

        tracking_obs = np.concatenate((sample_user_command, obs[:, :proprioceptive_sensor_dim]), axis=1)
        tracking_obs = env.force_normalize_observation(tracking_obs, type=2)
        with torch.no_grad():
            tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))
        rewards, dones = env.partial_step(tracking_action.cpu().detach().numpy())

        coordinate_obs = env.coordinate_observe()  # coordinate after taking step

        # update P_col and coordinate for terminated environment
        current_done_envs = np.where(dones == 1)[0]
        counter_current_done_envs = Counter(current_done_envs)
        counter_previous_done_envs = Counter(previous_done_envs)
        new_done_envs = np.array(sorted((counter_current_done_envs - counter_previous_done_envs).elements())).astype(int)
        done_envs.update(new_done_envs)
        previous_done_envs = current_done_envs.copy()

        # sum reward
        reward_sum[new_done_envs] += rewards[new_done_envs]
        counter_total_envs = Counter(np.arange(num_envs))
        not_done_envs = np.array(sorted((counter_total_envs - counter_current_done_envs).elements())).astype(int)
        reward_sum[not_done_envs] += rewards[not_done_envs]

        # logging different types of reward
        env.reward_logging(cfg['environment']['n_rewards'] + 1)
        reward_trajectory[new_done_envs, int(step / command_period_steps), :] += env.reward_log[new_done_envs, :]
        reward_trajectory[not_done_envs, int(step / command_period_steps), :] += env.reward_log[not_done_envs, :]
        reward_w_coeff_trajectory[new_done_envs, int(step / command_period_steps), :] += env.reward_w_cpeff_log[new_done_envs, :]
        reward_w_coeff_trajectory[not_done_envs, int(step / command_period_steps), :] += env.reward_w_cpeff_log[not_done_envs, :]

        # sum done
        if update_time:
            ppo.step(value_obs=planning_obs, rews=reward_sum, dones=dones)
            done_envs_storage.update(np.array(list(done_envs)))

        # reset COM buffer for terminated environment
        COM_buffer.partial_reset(current_done_envs)

    # reset only terminated environment
    env.initialize_n_step()  # to reset in new position
    env.partial_reset(list(done_envs))

    # save coordinate before taking step to modify the labeled data
    coordinate_obs = env.coordinate_observe()
    obs, _ = env.observe(False)  # observation before taking step

    COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
    COM_buffer.update(COM_feature)

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

    env.force_update_ob_rms(planning_obs, type=1)
    planning_obs = env.force_normalize_observation(planning_obs, type=1).astype(np.float32)

    # take st step to get value obs
    ppo.update(actor_obs=planning_obs, value_obs=planning_obs, log_this_iteration=(update % 10 == 0) and logging, update=update)

    actor.distribution.enforce_minimum_std(torch.from_numpy((action_clipping_range[:, 1] * 0.05).astype(np.float32)).to(device))

    mean_reward_sum = np.mean(reward_sum)
    mean_done_sum = len(list(done_envs_storage)) / num_envs

    if logging:
        if update % 5 == 0:
            # reward logging (value & std)
            reward_trajectory_mean = np.mean(reward_trajectory, axis=0)  # (n_plan_steps, cfg['environment']['n_rewards'] + 1)
            reward_w_coeff_trajectory_mean = np.mean(reward_w_coeff_trajectory, axis=0)  # (n_plan_steps, cfg['environment']['n_rewards'] + 1)
            reward_trajectory_mean[:, -1] = reward_w_coeff_trajectory_mean[:, -1]  # log reward sum considering the reward coeff
            reward_mean = np.mean(reward_trajectory_mean, axis=0)  # average reward for each planning step
            reward_std = np.std(reward_w_coeff_trajectory_mean, axis=0)  # variance of rewards in single trajectory
            assert reward_mean.shape[0] == cfg['environment']['n_rewards'] + 1
            assert reward_std.shape[0] == cfg['environment']['n_rewards'] + 1
            ppo.reward_logging(reward_names, reward_mean)   # logging mean of each reward (w/o coeff) and mean of reward sum (w/ coeff)
            ppo.reward_std_logging(reward_names, reward_std)   # logging std of each reward (w/ coeff) and std of reward sum (w/ coeff)

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average reward sum: ", '{:0.10f}'.format(mean_reward_sum)))   # logging mean of reward sum (w/ coeff)
    print('{:<40} {:>6}'.format("average dones: ", '{:0.6f}'.format(mean_done_sum)))
    print('{:<40} {:>6}'.format("elapsed time: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_n_track_steps * n_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_n_track_steps * n_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(actor.distribution.std.cpu().detach().numpy())
    print('----------------------------------------------------\n')

