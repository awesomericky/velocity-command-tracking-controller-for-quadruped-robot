import torch
import numpy as np


class Time_correlated_command_sampler:
    """
    Sample time correlated command :

    Time coorelation factor is controlled with 'beta'.
    """
    def __init__(self, random_command_sampler, beta=0.1):
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
            modified_command = self.new_command * self.beta + self.old_command * (1 - self.beta)
        self.old_command = modified_command
        return modified_command

    def reset(self):
        self.old_command = None
        self.new_command = None


class Noisy_command_sampler:
    """
    Sample noisy command with time correlation:

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


class Action_planner:
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
        self.delta = 0.5 * torch.tensor([self.max_forward_vel - self.min_forward_vel, self.max_lateral_vel - self.min_lateral_vel, self.max_yaw_rate - self.min_yaw_rate])

        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.action_dim = action_dim

        self.a_hat = torch.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = torch.zeros((self.n_sample, self.n_horizon, self.action_dim))

    def sample(self):
        epsil = torch.normal(0.0, self.sigma, size=(self.n_sample, self.n_horizon, self.action_dim))
        epsil = self.delta * epsil

        for h in range(self.n_horizon):
            if h == 0:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_hat[h, :]
            else:
                self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
                                        + (1 - self.beta) * self.a_tilda[:, h - 1, :]
        return self.a_tilda

    def update(self, rewards):
        rewards = torch.from_numpy(rewards)
        probs = torch.exp(self.gamma * (rewards - torch.max(rewards)))
        probs /= torch.sum(probs) + 1e-10
        self.a_hat = torch.sum(self.a_tilda * probs.unsqueeze(-1).unsqueeze(-1), dim=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0]


class Modified_action_planner:
    """
    Modified implementation of zeroth order stochastic optimizer by Kahn et al., Nagabandi et al. [1, 2]

    [1] https://arxiv.org/abs/2002.05700
    [2] https://arxiv.org/abs/1909.11652

    Initial command trajectory for optimization is set with desired user command at every step.
    Updating the optimal command trajectory at every step is excluded.
    """
    def __init__(self, command_range, n_sample, n_horizon, sigma, beta, action_dim=3):
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
        self.beta = beta
        self.action_dim = action_dim

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))

    # def sample(self, user_command):
    #     """
    #
    #     :param user_command: (self.action_dim, )  type: numpy tensor
    #     :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
    #     """
    #     self.a_hat = np.tile(user_command[np.newaxis, :], (self.n_horizon, 1))
    #     epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample - 1, self.n_horizon, self.action_dim))
    #     epsil = self.delta * epsil
    #     epsil = np.concatenate((np.zeros((1, self.n_horizon, self.action_dim)), epsil), axis=0)  # (self.n_sample, self.n_horizon, self.action_dim)
    #
    #     for h in range(self.n_horizon):
    #         if h == 0:
    #             self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
    #                                     + (1 - self.beta) * self.a_hat[h, :]
    #         elif h == self.n_horizon - 1:
    #             self.a_tilda[:, h, :] = self.beta * (self.a_hat[h, :] + epsil[:, h, :]) \
    #                                     + (1 - self.beta) * self.a_tilda[:, h - 1, :]
    #         else:
    #             self.a_tilda[:, h, :] = self.beta * (self.a_hat[h + 1, :] + epsil[:, h, :]) \
    #                                     + (1 - self.beta) * self.a_tilda[:, h - 1, :]
    #     return self.a_tilda

    def sample(self, user_command):
        """

        :param user_command: (self.action_dim, )  type: numpy tensor
        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_hat = np.tile(user_command[np.newaxis, :], (self.n_horizon, 1))
        epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample - 1, self.action_dim))
        epsil = self.delta * epsil
        epsil = np.broadcast_to(epsil[:, np.newaxis, :], (self.n_sample - 1, self.n_horizon, self.action_dim))
        epsil = np.concatenate((np.zeros((1, self.n_horizon, self.action_dim)), epsil), axis=0)  # (self.n_sample, self.n_horizon, self.action_dim)

        self.a_tilda = epsil + self.a_hat
        self.a_tilda[:, :, 0] = np.clip(self.a_tilda[:, :, 0], a_min=self.min_forward_vel, a_max=self.max_forward_vel)
        self.a_tilda[:, :, 1] = np.clip(self.a_tilda[:, :, 1], a_min=self.min_lateral_vel, a_max=self.max_lateral_vel)
        self.a_tilda[:, :, 2] = np.clip(self.a_tilda[:, :, 2], a_min=self.min_yaw_rate, a_max=self.max_yaw_rate)

        return self.a_tilda

    def action(self, rewards, safe=False):
        """

        :param rewards: (self.n_sample,)  type: numpy tensor
        :return:  (self.action_dim, )  type: numpy tensor
        """
        if safe:
            return self.a_hat[0, :]
        max_idx = np.argmax(rewards)
        return self.a_tilda[max_idx, 0, :], max_idx