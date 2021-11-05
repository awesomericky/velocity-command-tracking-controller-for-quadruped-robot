import pdb

import matplotlib.pyplot as plt
import torch
import numpy as np


class Time_correlated_command_sampler:
    """
    Sample time correlated command (using soft update rule):

    Time coorelation factor is controlled with 'beta'.
    
    Larger 'beta' == Bigger time correlation
    """
    def __init__(self, random_command_sampler, beta=0.7):
        self.old_command = None
        self.new_command = None
        self.random_command_sampler = random_command_sampler
        self.beta = beta

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        self.new_command = self.random_sample()
        if isinstance(self.old_command, type(None)):
            modified_command = self.new_command
        else:
            modified_command = self.old_command * self.beta + self.new_command * (1 - self.beta)
        self.old_command = modified_command
        return modified_command

    def reset(self):
        self.old_command = None
        self.new_command = None


class Normal_time_correlated_command_sampler:
    """
    Sample time correlated command (using normal distribution):

    Time coorelation factor is controlled with 'sigma'.
    """
    def __init__(self, random_command_sampler, cfg_command):
        self.old_command = None
        self.random_command_sampler = random_command_sampler
        self.cfg_command = cfg_command
        self.max_sigma = 0.5 * np.array([cfg_command["forward_vel"]["max"] - cfg_command["forward_vel"]["min"],
                                         cfg_command["lateral_vel"]["max"] - cfg_command["lateral_vel"]["min"],
                                         cfg_command["yaw_rate"]["max"] - cfg_command["yaw_rate"]["min"]])

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        if isinstance(self.old_command, type(None)):
            modified_command = self.random_sample()
        else:
            sigma_scale = np.random.uniform(0, 1, (self.random_command_sampler.n_envs, 3))  # sample command std scale (uniform distribution)
            sigma = self.max_sigma * sigma_scale
            modified_command = np.random.normal(self.old_command, sigma)  # sample command (normal distribution)
            modified_command = np.clip(modified_command,
                                       [self.cfg_command["forward_vel"]["min"], self.cfg_command["lateral_vel"]["min"], self.cfg_command["yaw_rate"]["min"]],
                                       [self.cfg_command["forward_vel"]["max"], self.cfg_command["lateral_vel"]["max"], self.cfg_command["yaw_rate"]["max"]])
        self.old_command = modified_command
        return np.ascontiguousarray(modified_command).astype(np.float32)

    def reset(self):
        self.old_command = None


class Noisy_command_sampler:
    """
    Sample noisy command:

    Noise factor is controlled with 'sigma'.
    """
    def __init__(self, random_command_sampler, sigma=0.05):
        self.old_command = None
        self.new_command = None
        self.random_command_sampler = random_command_sampler
        self.sigma = sigma

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        if isinstance(self.old_command, type(None)):
            self.new_command = self.random_sample()
            modified_command = self.new_command
        else:
            command_noise = np.random.normal(0, self.sigma, size=self.new_command.shape).astype(np.float32)
            modified_command = self.new_command + command_noise
        self.old_command = modified_command
        return modified_command

    def reset(self):
        self.old_command = None
        self.new_command = None


class Command_sampler:
    """
    Sample constant command:

    """
    def __init__(self, random_command_sampler):
        self.new_command = None
        self.random_command_sampler = random_command_sampler

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        if isinstance(self.new_command, type(None)):
            self.new_command = self.random_sample()
        return self.new_command

    def reset(self):
        self.new_command = None


class Zeroth_action_planner:
    """
    Implementation of zeroth order stochastic optimizer by Kahn et al., Nagabandi et al. [1, 2]

    [1] https://arxiv.org/abs/2002.05700
    [2] https://arxiv.org/abs/1909.11652
    """
    def __init__(self, command_range, n_sample, n_horizon, sigma, gamma, beta, action_dim=3):
        """

        :param command_limit: available command range
        :param n_sample: number of sampling trajectory
        :param n_horizon: number of predicted future steps
        :param sigma: std of new action sampling distribution
        :param gamma: factor that determines the degree for high reward action
        :param beta: factor that determines the degree of action time correlation
        """
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.delta = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel, self.max_lateral_vel - self.min_lateral_vel, self.max_yaw_rate - self.min_yaw_rate])

        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.action_dim = action_dim

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))

    def sample(self):
        epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample, self.n_horizon, self.action_dim))
        epsil = self.delta * epsil

        for h in range(self.n_horizon):
            if h == 0:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_hat[h, :]
            elif h == self.n_horizon - 1:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_tilda[:, h - 1, :]
            else:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_tilda[:, h - 1, :]

            self.a_tilda[:, h, :] = np.clip(self.a_tilda[:, h, :],
                                            a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                            a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def update(self, rewards):
        probs = np.exp(self.gamma * (rewards - np.max(rewards)))
        probs = np.exp(self.gamma * rewards)
        probs /= (np.sum(probs) + 1e-10)
        self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0], self.a_hat.astype(np.float32)

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))


class Modified_zeroth_action_planner:
    """
    Modified implementation of zeroth order stochastic optimizer by Kahn et al., Nagabandi et al. [1, 2]
    Constant action for each trajectory.

    [1] https://arxiv.org/abs/2002.05700
    [2] https://arxiv.org/abs/1909.11652
    """
    def __init__(self, command_range, n_sample, n_horizon, sigma, gamma, beta, noise_sigma=0.1, noise=False, action_dim=3):
        """

        :param command_limit: available command range
        :param n_sample: number of sampling trajectory
        :param n_horizon: number of predicted future steps
        :param sigma: std of new action sampling distribution
        :param gamma: factor that determines the degree for high reward action
        :param beta: factor that determines the degree of action time correlation
        """
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.sigma = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel,
                                     self.max_lateral_vel - self.min_lateral_vel,
                                     self.max_yaw_rate - self.min_yaw_rate])

        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.noise_sigma = noise_sigma
        self.noise = noise
        self.action_dim = action_dim

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))

    def sample(self):
        epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample, self.action_dim))
        epsil = self.sigma * epsil  # same as sampling from ~ N(0, self.sigma)
        epsil = np.broadcast_to(epsil[:, np.newaxis, :], (self.n_sample, self.n_horizon, self.action_dim)).copy()

        for h in range(self.n_horizon):
            if h == 0:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_hat[h, :]
            elif h == self.n_horizon - 1:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_tilda[:, h - 1, :]
            else:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_tilda[:, h - 1, :]

            self.a_tilda[:, h, :] = np.clip(self.a_tilda[:, h, :],
                                            a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                            a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        if self.noise:
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            self.a_tilda[:, 1:, :] += noise_epsil

        self.a_tilda = np.clip(self.a_tilda,
                                a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def update(self, rewards):
        probs = np.exp(self.gamma * (rewards - np.max(rewards)))
        probs /= np.sum(probs) + 1e-10
        self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0], self.a_hat.astype(np.float32)

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))


class Stochastic_action_planner_uniform_bin:
    """
    Sample commands from uniformly seperated_bin
    """
    def __init__(self, command_range, n_sample, n_horizon, n_bin, beta, gamma, noise_sigma=0.1, noise=False, action_dim=3):
        """

        :param command_range: available command range
        :param n_sample: number of sampling trajectory
        :param n_horizon: number of predicted future steps
        :param n_bin: number of seperate bins for available command range
        :param beta: factor that determines the degree of action time correlation
        :param noise_sigma: std of noise distribution added to each time step
        :param noise: whether to give noise or not to across different time steps
        """
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.forward_vel_bin_size = (self.max_forward_vel - self.min_forward_vel) / n_bin
        self.forward_vel_bin_array = np.arange(self.min_forward_vel, self.max_forward_vel, self.forward_vel_bin_size)
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.lateral_vel_bin_size = (self.max_lateral_vel - self.min_lateral_vel) / n_bin
        self.lateral_vel_bin_array = np.arange(self.min_lateral_vel, self.max_lateral_vel, self.lateral_vel_bin_size)
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.yaw_rate_bin_size = (self.max_yaw_rate - self.min_yaw_rate) / n_bin
        self.yaw_rate_bin_array = np.arange(self.min_yaw_rate, self.max_yaw_rate, self.yaw_rate_bin_size)
        self.noise = noise

        self.n_sample = n_sample
        self.n_bin = n_bin
        self.beta = beta
        self.gamma = gamma
        self.n_horizon = n_horizon
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.action_bottom_margin = np.zeros((self.n_sample, self.action_dim))
        for i in range(self.n_sample):
            forward_vel_idx = (i // (self.n_bin ** 2)) % self.n_bin
            lateral_vel_idx = (i // self.n_bin) % self.n_bin
            yaw_rate_idx = i % self.n_bin
            self.action_bottom_margin[i] = [self.forward_vel_bin_array[forward_vel_idx],
                                            self.lateral_vel_bin_array[lateral_vel_idx],
                                            self.yaw_rate_bin_array[yaw_rate_idx]]

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def sample(self):
        """

        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_tilda = np.random.uniform(0.0, 1.0, size=(self.n_sample, self.action_dim))
        self.a_tilda[:, 0] *= self.forward_vel_bin_size
        self.a_tilda[:, 1] *= self.lateral_vel_bin_size
        self.a_tilda[:, 2] *= self.yaw_rate_bin_size
        self.a_tilda += self.action_bottom_margin
        self.a_tilda = self.a_tilda[:, np.newaxis, :]
        self.a_tilda = np.broadcast_to(self.a_tilda, (self.n_sample, self.n_horizon, self.action_dim)).copy()

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda = self.a_tilda * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        # add noise
        if self.noise:
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            self.a_tilda[:, 1:, :] += noise_epsil

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        # probs = np.exp(self.gamma * rewards)
        # probs /= np.sum(probs) + 1e-10
        # self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)
        safe_idx = np.where(rewards != 0)[0]
        # safe_idx = np.argpartition(rewards, -500)[-500:]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= (np.sum(probs) + 1e-10)
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        if self.noise:
            a_hat_traj = np.broadcast_to(self.a_hat[0], (self.n_horizon, self.action_dim)).copy()
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_horizon - 1, self.action_dim))
            a_hat_traj[1:, :] += noise_epsil
        else:
            a_hat_traj = np.tile(self.a_hat[0], (self.n_horizon, 1))
        # max_idx = np.argmax(rewards)
        # return self.a_tilda[max_idx, 0, :].astype(np.float32), self.a_tilda[max_idx, :, :].astype(np.float32)
        return self.a_hat[0], a_hat_traj.astype(np.float32)


class Stochastic_action_planner_uniform_bin_w_time_correlation:
    """
    Sample commands from uniformly seperated_bin
    """
    def __init__(self, command_range, n_sample, n_horizon, n_bin, beta, gamma,
                 random_command_sampler, time_correlation_beta=0.1,
                 noise_sigma=0.1, noise=False, action_dim=3):
        """

        :param command_range: available command range
        :param n_sample: number of sampling trajectory
        :param n_horizon: number of predicted future steps
        :param n_bin: number of seperate bins for available command range
        :param beta: factor that determines the degree of action time correlation
        :param noise_sigma: std of noise distribution added to each time step
        :param noise: whether to give noise or not to across different time steps
        """
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.forward_vel_bin_size = (self.max_forward_vel - self.min_forward_vel) / n_bin
        self.forward_vel_bin_array = np.arange(self.min_forward_vel, self.max_forward_vel, self.forward_vel_bin_size)
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.lateral_vel_bin_size = (self.max_lateral_vel - self.min_lateral_vel) / n_bin
        self.lateral_vel_bin_array = np.arange(self.min_lateral_vel, self.max_lateral_vel, self.lateral_vel_bin_size)
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.yaw_rate_bin_size = (self.max_yaw_rate - self.min_yaw_rate) / n_bin
        self.yaw_rate_bin_array = np.arange(self.min_yaw_rate, self.max_yaw_rate, self.yaw_rate_bin_size)
        self.noise = noise

        self.n_sample = n_sample
        self.n_bin = n_bin
        self.beta = beta
        self.gamma = gamma
        self.n_horizon = n_horizon
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.action_bottom_margin = np.zeros((self.n_sample, self.action_dim))
        for i in range(self.n_sample):
            forward_vel_idx = (i // (self.n_bin ** 2)) % self.n_bin
            lateral_vel_idx = (i // self.n_bin) % self.n_bin
            yaw_rate_idx = i % self.n_bin
            self.action_bottom_margin[i] = [self.forward_vel_bin_array[forward_vel_idx],
                                            self.lateral_vel_bin_array[lateral_vel_idx],
                                            self.yaw_rate_bin_array[yaw_rate_idx]]

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

        self.random_command_sampler = random_command_sampler
        self.time_correlation_beta = time_correlation_beta

    def sample(self):
        """

        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_tilda = np.random.uniform(0.0, 1.0, size=(self.n_sample, self.action_dim))
        self.a_tilda[:, 0] *= self.forward_vel_bin_size
        self.a_tilda[:, 1] *= self.lateral_vel_bin_size
        self.a_tilda[:, 2] *= self.yaw_rate_bin_size
        self.a_tilda += self.action_bottom_margin
        self.a_tilda = self.a_tilda[:, np.newaxis, :]
        self.a_tilda = np.broadcast_to(self.a_tilda, (self.n_sample, self.n_horizon, self.action_dim)).copy()

        for i in range(1, self.n_horizon):
            random_command = self.random_command_sampler.uniform_sample_train()
            self.a_tilda[:, i, :] = self.time_correlation_beta * random_command + (1 - self.time_correlation_beta) * self.a_tilda[:, i-1, :]

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda = self.a_tilda * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        # add noise
        if self.noise:
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            self.a_tilda[:, 1:, :] += noise_epsil

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        # probs = np.exp(self.gamma * rewards)
        # probs /= np.sum(probs) + 1e-10
        # self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)
        safe_idx = np.where(rewards != 0)[0]
        # safe_idx = np.argpartition(rewards, -500)[-500:]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0], self.a_hat.astype(np.float32)


class Stochastic_action_planner_uniform_bin_w_time_correlation_nprmal:
    def __init__(self, command_range, n_sample, n_horizon, n_bin, beta, gamma,
                 random_command_sampler, time_correlation_beta=0.1,
                 noise_sigma=0.1, noise=False, action_dim=3):
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.forward_vel_bin_size = (self.max_forward_vel - self.min_forward_vel) / n_bin
        self.forward_vel_bin_array = np.arange(self.min_forward_vel, self.max_forward_vel, self.forward_vel_bin_size)
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.lateral_vel_bin_size = (self.max_lateral_vel - self.min_lateral_vel) / n_bin
        self.lateral_vel_bin_array = np.arange(self.min_lateral_vel, self.max_lateral_vel, self.lateral_vel_bin_size)
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.yaw_rate_bin_size = (self.max_yaw_rate - self.min_yaw_rate) / n_bin
        self.yaw_rate_bin_array = np.arange(self.min_yaw_rate, self.max_yaw_rate, self.yaw_rate_bin_size)
        self.noise = noise

        self.max_sigma = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel,
                                         self.max_lateral_vel - self.min_lateral_vel,
                                         self.max_yaw_rate - self.min_yaw_rate])

        self.n_sample = n_sample
        self.n_bin = n_bin
        self.beta = beta
        self.gamma = gamma
        self.n_horizon = n_horizon
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.action_bottom_margin = np.zeros((self.n_sample, self.action_dim))
        for i in range(self.n_sample):
            forward_vel_idx = (i // (self.n_bin ** 2)) % self.n_bin
            lateral_vel_idx = (i // self.n_bin) % self.n_bin
            yaw_rate_idx = i % self.n_bin
            self.action_bottom_margin[i] = [self.forward_vel_bin_array[forward_vel_idx],
                                            self.lateral_vel_bin_array[lateral_vel_idx],
                                            self.yaw_rate_bin_array[yaw_rate_idx]]

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

        self.random_command_sampler = random_command_sampler
        self.time_correlation_beta = time_correlation_beta

    def sample(self):
        """

        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_tilda = np.random.uniform(0.0, 1.0, size=(self.n_sample, self.action_dim))
        self.a_tilda[:, 0] *= self.forward_vel_bin_size
        self.a_tilda[:, 1] *= self.lateral_vel_bin_size
        self.a_tilda[:, 2] *= self.yaw_rate_bin_size
        self.a_tilda += self.action_bottom_margin
        self.a_tilda = self.a_tilda[:, np.newaxis, :]
        self.a_tilda = np.broadcast_to(self.a_tilda, (self.n_sample, self.n_horizon, self.action_dim)).copy()

        for i in range(1, self.n_horizon):
            sigma_scale = np.random.uniform(0, 0.5, (self.random_command_sampler.n_envs, 3))  # sample command std scale (uniform distribution)
            sigma = self.max_sigma * sigma_scale
            self.a_tilda[:, i, :] = np.random.normal(self.a_tilda[:, i-1, :], sigma)  # sample command (normal distribution)
            self.a_tilda[:, i, :] = np.clip(self.a_tilda[:, i, :],
                                            a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                            a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda = self.a_tilda * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        # add noise
        if self.noise:
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            self.a_tilda[:, 1:, :] += noise_epsil

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        # probs = np.exp(self.gamma * rewards)
        # probs /= np.sum(probs) + 1e-10
        # self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)
        safe_idx = np.where(rewards != 0)[0]
        # safe_idx = np.argpartition(rewards, -500)[-500:]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0], self.a_hat.astype(np.float32)


class Stochastic_action_planner_normal:
    """
    Sample commands from normal distribution, where the mean value is user command
    """
    def __init__(self, command_range, n_sample, n_horizon, sigma, beta, gamma, noise_sigma=0.1, noise=True, action_dim=3):
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.delta = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel, self.max_lateral_vel - self.min_lateral_vel, self.max_yaw_rate - self.min_yaw_rate])
        self.noise = noise

        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.sigma = sigma
        self.gamma = gamma
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))
        self.first = True
        self.beta = beta

    def sample(self, user_command):
        """

        :param user_command: (self.action_dim, )  type: numpy tensor
        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample - 1, self.action_dim))
        epsil = self.delta * epsil
        epsil = np.broadcast_to(epsil[:, np.newaxis, :], (self.n_sample - 1, self.n_horizon, self.action_dim)).copy()
        epsil = np.concatenate((np.zeros((1, self.n_horizon, self.action_dim)), epsil), axis=0)  # (self.n_sample, self.n_horizon, self.action_dim)

        if self.noise:
            # add extra noise along the command trajectory
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            epsil[:, 1:, :] += noise_epsil

        self.a_tilda = epsil + user_command

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda[1:, :, :] = self.a_tilda[1:, :, :] * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        safe_idx = np.where(rewards != 0)[0]
        # safe_idx = np.argpartition(rewards, -500)[-500:]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards, safe=False):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        if safe:
            return self.a_tilda[0, 0, :]
        self.update(rewards)
        if self.noise:
            noise_epsil = np.concatenate((np.zeros((1, self.action_dim)), np.random.normal(0.0, self.noise_sigma, size=(self.n_horizon - 1, self.action_dim))), axis=0)
            a_hat_traj = self.a_hat[0, :] + noise_epsil
        else:
            a_hat_traj = np.broadcast_to(self.a_hat[0], (self.n_horizon, self.action_dim)).copy()
        return self.a_hat[0], a_hat_traj.astype(np.float32)


class Stochastic_action_planner_uniform_bin_baseline:
    """
    Sample commands from uniformly seperated_bin
    """
    def __init__(self, command_range, n_sample, n_horizon, n_bin, beta, gamma, action_dim=3):
        """

        :param command_range: available command range
        :param n_sample: number of sampling trajectory
        :param n_horizon: number of predicted future steps
        :param n_bin: number of seperate bins for available command range
        :param beta: factor that determines the degree of action time correlation
        :param noise_sigma: std of noise distribution added to each time step
        :param noise: whether to give noise or not to across different time steps
        """
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.forward_vel_bin_size = (self.max_forward_vel - self.min_forward_vel) / n_bin
        self.forward_vel_bin_array = np.arange(self.min_forward_vel, self.max_forward_vel, self.forward_vel_bin_size)
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.lateral_vel_bin_size = (self.max_lateral_vel - self.min_lateral_vel) / n_bin
        self.lateral_vel_bin_array = np.arange(self.min_lateral_vel, self.max_lateral_vel, self.lateral_vel_bin_size)
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.yaw_rate_bin_size = (self.max_yaw_rate - self.min_yaw_rate) / n_bin
        self.yaw_rate_bin_array = np.arange(self.min_yaw_rate, self.max_yaw_rate, self.yaw_rate_bin_size)

        self.n_sample = n_sample
        self.n_bin = n_bin
        self.beta = beta
        self.gamma = gamma
        self.n_horizon = n_horizon
        self.action_dim = action_dim

        self.action_bottom_margin = np.zeros((self.n_sample, self.action_dim))
        for i in range(self.n_sample):
            forward_vel_idx = (i // (self.n_bin ** 2)) % self.n_bin
            lateral_vel_idx = (i // self.n_bin) % self.n_bin
            yaw_rate_idx = i % self.n_bin
            self.action_bottom_margin[i] = [self.forward_vel_bin_array[forward_vel_idx],
                                            self.lateral_vel_bin_array[lateral_vel_idx],
                                            self.yaw_rate_bin_array[yaw_rate_idx]]

        self.a_hat = np.zeros(self.action_dim)
        self.first = True

    def sample(self):
        """

        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_tilda = np.random.uniform(0.0, 1.0, size=(self.n_sample, self.action_dim))
        self.a_tilda[:, 0] *= self.forward_vel_bin_size
        self.a_tilda[:, 1] *= self.lateral_vel_bin_size
        self.a_tilda[:, 2] *= self.yaw_rate_bin_size
        self.a_tilda += self.action_bottom_margin

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda = self.a_tilda * (1 - self.beta) + self.a_hat[np.newaxis, :] * self.beta

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros(self.action_dim)
        self.first = True

    def update(self, rewards):
        # probs = np.exp(self.gamma * rewards)
        # probs /= np.sum(probs) + 1e-10
        # self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis], axis=0)

        safe_idx = np.where(rewards != 0)[0]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :] * probs[:, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return: (self.action_dim, )  type: numpy
        """
        self.update(rewards)
        return self.a_hat.copy()
