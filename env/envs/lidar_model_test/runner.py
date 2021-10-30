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
from raisimGymTorch.env.envs.lidar_model.trainer import Trainer
from raisimGymTorch.env.envs.lidar_model.action import Noisy_command_sampler
from raisimGymTorch.env.envs.lidar_model.storage import Buffer


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

# user command samping
user_command = UserCommand(cfg, cfg['environment']['num_envs'])
command_sampler = Noisy_command_sampler(user_command, cfg["data_collection"]["command_sampler_sigma"])

# create environment from the configuration file
env = VecEnv(lidar_model.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['data_collection']['command_period'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

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

environment_model = Lidar_environment_model(lidar_encoding_config=cfg["architecture"]["state_encoder"],
                                            command_encoding_config=cfg["architecture"]["command_encoder"],
                                            recurrence_config=cfg["architecture"]["recurrence"],
                                            prediction_config=cfg["architecture"]["traj_predictor"],
                                            device=device)

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
                  device=device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

# wandb initialize
wandb.init(name=task_name, project="Quadruped_RL")

if mode == 'retrain':
    load_enviroment_model_param(weight_path, environment_model, trainer.optimizer, saver.data_dir)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['architecture']['command_tracking_policy_net'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))


for update in range(cfg["environment"]["max_n_update"]):
    start = time.time()

    if update % cfg["environment"]["eval_every_n"] == 0:
        print("Evaluating the current environment model")
        torch.save({
            'model_architecture_state_dict': environment_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')

        # we create another graph just to demonstrate the save/load method
        loaded_environment_model = Lidar_environment_model(lidar_encoding_config=cfg["architecture"]["state_encoder"],
                                                           command_encoding_config=cfg["architecture"]["command_encoder"],
                                                           recurrence_config=cfg["architecture"]["recurrence"],
                                                           prediction_config=cfg["architecture"]["traj_predictor"],
                                                           device=device)
        loaded_environment_model.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['model_architecture_state_dict'])
        loaded_environment_model.eval()
        loaded_environment_model.to(device)

        env.initialize_n_step()
        env.reset()
        env.turn_on_visualization()
        command_sampler.reset()
        COM_buffer.reset()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        state_traj = []
        command_traj = []
        P_col_traj = []
        coordinate_traj = []
        init_coordinate_traj = []
        done_envs = set()

        assert n_steps % command_period_steps == 0, "Total steps in training should be divided by command period steps."

        for step in range(n_steps):
            frame_start = time.time()
            new_command_time = step % command_period_steps == 0
            traj_update_time = (step + 1) % command_period_steps == 0

            if new_command_time:
                # reset only terminated environment
                env.partial_reset(list(done_envs))

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
                temp_command = np.zeros((cfg['environment']['num_envs'], command_dim))
                temp_P_col = np.zeros(cfg['environment']['num_envs'])
                temp_coordinate = np.zeros((cfg['environment']['num_envs'], coordinate_dim))

                lidar_data = obs[:, proprioceptive_sensor_dim:]
                COM_history_feature = COM_buffer.return_data(flatten=True)
                temp_state = np.concatenate((lidar_data, COM_history_feature), axis=1)

                # if step % 300 == 0:
                #     command_sampler.reset()

                sample_user_command = command_sampler.sample()
                temp_command = sample_user_command


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
        (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (mean_P_col_accuracy, mean_coordinate_error) \
            = trainer.evaluate(environment_model=loaded_environment_model,
                               state_traj=state_traj,
                               command_traj=command_traj,
                               dones_traj=P_col_traj,
                               coordinate_traj=coordinate_traj,
                               init_coordinate_traj=init_coordinate_traj)

        plot_trajectory_prediction_result(P_col_traj=np.swapaxes(real_P_cols, 0, 1),
                                          coordinate_traj=np.swapaxes(real_coordinates, 0, 1),
                                          predicted_P_col_traj=np.swapaxes(predicted_P_cols, 0, 1),
                                          predicted_coordinate_traj=np.swapaxes(predicted_coordinates, 0, 1),
                                          task_name=saver.data_dir.split('/')[-2],
                                          run_name=saver.data_dir.split('/')[-1],
                                          n_update=update)

        print('====================================================')
        print('{:>6}th evaluation'.format(update))
        print('{:<40} {:>6}'.format("collision accuracy: ", '{:0.6f}'.format(mean_P_col_accuracy)))
        print('{:<40} {:>6}'.format("coordinate error: ", '{:0.6f}'.format(mean_coordinate_error)))

        print('====================================================\n')

        # env.stop_video_recording()
        env.turn_off_visualization()

    # prepare for training
    env.initialize_n_step()
    env.reset()
    command_sampler.reset()
    COM_buffer.reset()

    state_traj = []
    command_traj = []
    P_col_traj = []
    coordinate_traj = []
    init_coordinate_traj = []
    done_envs = set()

    for step in range(n_steps):
        frame_start = time.time()
        new_command_time = step % command_period_steps == 0
        traj_update_time = (step + 1) % command_period_steps == 0

        if new_command_time:
            # reset only terminated environment
            env.partial_reset(list(done_envs))

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
            temp_command = np.zeros((cfg['environment']['num_envs'], command_dim))
            temp_P_col = np.zeros(cfg['environment']['num_envs'])
            temp_coordinate = np.zeros((cfg['environment']['num_envs'], coordinate_dim))

            lidar_data = obs[:, proprioceptive_sensor_dim:]
            COM_history_feature = COM_buffer.return_data(flatten=True)
            temp_state = np.concatenate((lidar_data, COM_history_feature), axis=1)

            # if step % 300 == 0:
            #     command_sampler.reset()

            sample_user_command = command_sampler.sample()
            temp_command = sample_user_command


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

    # update training data buffer
    state_traj = np.array(state_traj)
    command_traj = np.array(command_traj)
    P_col_traj = np.array(P_col_traj)
    coordinate_traj = np.array(coordinate_traj)
    init_coordinate_traj = np.array(init_coordinate_traj)
    trainer.update_data(state_traj=state_traj,
                        command_traj=command_traj,
                        dones_traj=P_col_traj,
                        coordinate_traj=coordinate_traj,
                        init_coordinate_traj=init_coordinate_traj)

    mean_loss, mean_P_col_loss, mean_coordinate_loss = 0.0, 0.0, 0.0
    if trainer.is_buffer_full():
        mean_loss, mean_P_col_loss, mean_coordinate_loss = trainer.train()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("loss: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("collision loss: ", '{:0.6f}'.format(mean_P_col_loss)))
    print('{:<40} {:>6}'.format("coordinate loss: ", '{:0.6f}'.format(mean_coordinate_loss)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')




