from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import anymal_rough
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
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# task specification
task_name = "anymal_locomotion_rough"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
reward_names = list(map(str, cfg['environment']['reward'].keys()))
reward_names.append('reward_sum')

# user command sampling
user_command = UserCommand(cfg, cfg['environment']['num_envs'])

# create environment from the configuration file
env = VecEnv(anymal_rough.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
command_period_steps = math.floor(cfg['environment']['command_period'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, 1.0),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

# tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update

# wandb initialize
wandb.init(name=task_name, project="Quadruped_RL")

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.9988,  # discount factor
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

for update in range(20000):
    start = time.time()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.initialize_n_step()
        env.reset()
        env.turn_on_visualization()
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        command_trajectory = []
        real_trajectory = []

        for step in range(n_steps*2):
            if step % command_period_steps == 0:
                sample_user_command = user_command.uniform_sample_evaluate()
                # sample_user_command[:, 2] = 0  # set yaw rate command to zero
                env.set_user_command(sample_user_command)   # Hash this when n_env=1 for logging

            frame_start = time.time()
            obs, non_obs = env.observe(False)
            action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
            reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)

            command_trajectory.append(sample_user_command[0])
            real_trajectory.append([non_obs[0, 18], non_obs[0, 19], non_obs[0, 23]])

            if wait_time > 0.:
                time.sleep(wait_time)

        command_trajectory = np.array(command_trajectory)
        real_trajectory = np.array(real_trajectory)
        plot_command_tracking_result(command_trajectory, real_trajectory, saver.data_dir.split('/')[-2], saver.data_dir.split('/')[-1], update, control_dt=cfg['environment']['control_dt'])

        env.stop_video_recording()
        env.stop_video_recording()
        env.turn_off_visualization()

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    env.initialize_n_step()
    env.reset()
    reward_trajectory = np.zeros((cfg['environment']['num_envs'], n_steps, cfg['environment']['n_rewards'] + 1))

    # actual training
    for step in range(n_steps):
        if step % command_period_steps == 0:
            sample_user_command = user_command.uniform_sample_train()
            # sample_user_command[:, 2] = 0  # set yaw rate command to zero
            env.set_user_command(sample_user_command)

        obs, _ = env.observe()
        action = ppo.observe(obs)
        reward, dones = env.step(action)
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + sum(dones)
        reward_ll_sum = reward_ll_sum + sum(reward)

        env.reward_logging(cfg['environment']['n_rewards'] + 1)
        reward_trajectory[:, step, :] = env.reward_log

    # take st step to get value obs
    obs, _ = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

    # reward logging (value & std)
    if update % 5 == 0:
        reward_trajectory_mean = np.mean(reward_trajectory, axis=0)  # (n_steps, cfg['environment']['n_rewards'] + 1)
        reward_mean = np.mean(reward_trajectory_mean, axis=0)
        reward_std = np.std(reward_trajectory_mean, axis=0)
        assert reward_mean.shape[0] == cfg['environment']['n_rewards'] + 1
        assert reward_std.shape[0] == cfg['environment']['n_rewards'] + 1
        ppo.reward_logging(reward_names, reward_mean)
        ppo.reward_std_logging(reward_names, reward_std)

    # curriculum learning
    env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('std: ')
    print(np.exp(actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')
