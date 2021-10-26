from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import anymal_flat_model
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import UserCommand
from raisimGymTorch.helper.utils_plot import plot_command_tracking_result
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse
import numpy as np
import datetime
import pdb


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# user command samping
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

# # create environment from the configuration file
# cfg['environment']['num_envs'] = 1

try:
    cfg['environment']['num_threads'] = cfg['environment']['test_num_threads']
except:
    pass

env = VecEnv(anymal_flat_model.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
ob_dim = env.num_obs + 3  # command_dim : 3
act_dim = env.num_acts

command_period_steps = math.floor(cfg['environment']['command_period'] / cfg['environment']['control_dt'])

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path)['actor_architecture_state_dict'])
    loaded_graph.to(device)

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()
    # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_test"+'.mp4')

    # max_steps = 1000000
    max_steps = 3000 ## 30 secs
    command_trajectory = []
    real_trajectory = []
    prediction_step = 600

    for step in range(max_steps):
        frame_start = time.time()
        if step % command_period_steps == 0:
            sample_user_command = user_command.uniform_sample_train()

            observation_time = []
            observation_processing_time = []
            policy_time = []
            env_step_time = []
            coordinate_observation_time = []

            rollout_start = time.time()
            for i in range(prediction_step):
                roll_step1 = time.time()
                obs, non_obs = env.observe(False)
                roll_step2 = time.time()
                obs = np.concatenate((sample_user_command, obs), axis=1)
                obs = env.force_normalize_observation(obs, type=1)
                roll_step3 = time.time()
                # action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
                action_ll = loaded_graph.architecture(torch.from_numpy(obs).to(device))
                roll_step4 = time.time()
                _, dones = env.step(action_ll.cpu().detach().numpy())
                roll_step5 = time.time()
                coordinate_obs = env.coordinate_observe()
                roll_step6 = time.time()

                observation_time.append(roll_step2 - roll_step1)
                observation_processing_time.append(roll_step3 - roll_step2)
                policy_time.append(roll_step4 - roll_step3)
                env_step_time.append(roll_step5 - roll_step4)
                coordinate_observation_time.append(roll_step6 - roll_step5)

                # print('----------------------------------------------------')
                # print('{:>6}th step'.format(i))
                # print('{:<40} {:>6}'.format("observation: ", '{:0.10f}'.format(roll_step2 - roll_step1)))
                # print('{:<40} {:>6}'.format("observation processing: ", '{:0.10f}'.format(roll_step3 - roll_step2)))
                # print('{:<40} {:>6}'.format("policy: ", '{:0.10f}'.format(roll_step4 - roll_step3)))
                # print('{:<40} {:>6}'.format("env step: ", '{:0.10f}'.format(roll_step5 - roll_step4)))
                # print('{:<40} {:>6}'.format("coordinate observation: ", '{:0.10f}'.format(roll_step6 - roll_step5)))
                # print('----------------------------------------------------\n')

                # assert dones.any() is not True, "Environment finished."
            rollout_end = time.time()
            print(rollout_end - rollout_start)

            observation_time = np.array(observation_time)
            observation_processing_time = np.array(observation_processing_time)
            policy_time = np.array(policy_time)
            env_step_time = np.array(env_step_time)
            coordinate_observation_time = np.array(coordinate_observation_time)

            print('----------------------------------------------------')
            print('{:>6}th step'.format(i))
            print('[Mean]')
            print('{:<40} {:>6}'.format("observation: ", '{:0.10f}'.format(np.mean(observation_time))))
            print('{:<40} {:>6}'.format("observation processing: ", '{:0.10f}'.format(np.mean(observation_processing_time))))
            print('{:<40} {:>6}'.format("policy: ", '{:0.10f}'.format(np.mean(policy_time))))
            print('{:<40} {:>6}'.format("env step: ", '{:0.10f}'.format(np.mean(env_step_time))))
            print('{:<40} {:>6}'.format("coordinate observation: ", '{:0.10f}'.format(np.mean(coordinate_observation_time))))
            # print("\n")
            # print('[std]')
            # print('{:<40} {:>6}'.format("observation: ", '{:0.10f}'.format(np.std(observation_time))))
            # print('{:<40} {:>6}'.format("observation processing: ", '{:0.10f}'.format(np.std(observation_processing_time))))
            # print('{:<40} {:>6}'.format("policy: ", '{:0.10f}'.format(np.std(policy_time))))
            # print('{:<40} {:>6}'.format("env step: ", '{:0.10f}'.format(np.std(env_step_time))))
            # print('{:<40} {:>6}'.format("coordinate observation: ", '{:0.10f}'.format(np.std(coordinate_observation_time))))
            print('----------------------------------------------------\n')

        env.reset()
        obs, non_obs = env.observe(False)
        obs = np.concatenate((sample_user_command, obs), axis=1)
        obs = env.force_normalize_observation(obs, type=1)
        # action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).to(device))
        _, dones = env.step(action_ll.cpu().detach().numpy())
        coordinate_obs = env.coordinate_observe()

        if dones.all():
            env.reset()

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

        if wait_time > 0.:
            time.sleep(wait_time)

    # for step in range(max_steps):
    #     frame_start = time.time()
    #     if step % command_period_steps == 0:
    #         sample_user_command = user_command.uniform_sample_train()
    #
    #     obs, non_obs = env.observe(False)
    #     obs = np.concatenate((sample_user_command, obs), axis=1)
    #     obs = env.force_normalize_observation(obs, type=1)
    #     action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
    #     _, dones = env.step(action_ll.cpu().detach().numpy())
    #     coordinate_obs = env.coordinate_observe()
    #
    #     if dones:
    #         env.reset()
    #
    #     frame_end = time.time()
    #     wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
    #
    #     if wait_time > 0.:
    #         time.sleep(wait_time)
    #
    #     # # command tracking logging
    #     # command_trajectory.append(sample_user_command[0])
    #     # real_trajectory.append([non_obs[0, 15], non_obs[0, 16], non_obs[0, 20]])
    
    env.turn_off_visualization()
    env.stop_video_recording()

    # command_trajectory = np.array(command_trajectory)
    # real_trajectory = np.array(real_trajectory)
    #
    # plot_command_tracking_result(command_trajectory, real_trajectory, weight_path.split('/')[-3], weight_path.split('/')[-2], 'test1010', control_dt=cfg['environment']['control_dt'])

    print("Finished at the maximum visualization steps")
