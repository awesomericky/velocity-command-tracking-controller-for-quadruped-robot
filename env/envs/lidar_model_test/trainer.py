from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .storage import DataStorage, Buffer
import pdb
import wandb


class Trainer:
    def __init__(self,
                 environment_model,
                 state_dim,
                 command_dim,
                 P_col_dim,
                 coordinate_dim,
                 prediction_period=3,  # [s]
                 delta_prediction_time=0.5,  # [s]
                 loss_weight=None,
                 max_storage_size=1280,
                 num_learning_epochs=10,
                 mini_batch_size=64,
                 device='cpu',
                 shuffle_batch=True,
                 clip_grad=False,
                 learning_rate=5e-4,
                 max_grad_norm=0.5):

        self.n_prediction_step = int(prediction_period / delta_prediction_time)
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.P_col_dim = P_col_dim
        self.coordinate_dim = coordinate_dim

        # environment model
        self.environment_model = environment_model
        self.environment_model.train()
        self.environment_model.to(device)

        self.storage = DataStorage(max_storage_size=max_storage_size,
                                   state_dim=state_dim,
                                   command_shape=[self.n_prediction_step, command_dim],
                                   P_col_shape=[self.n_prediction_step, P_col_dim],
                                   coordinate_shape=[self.n_prediction_step, coordinate_dim],
                                   device=device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.environment_model.parameters()], lr=learning_rate)
        self.device = device

        # training hyperparameters
        self.max_storage_size = max_storage_size
        self.num_learning_epochs = num_learning_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm

        if loss_weight is None:
            self.loss_weight = {"collision": 1, "coordinate": 1}
        else:
            assert isinstance(loss_weight, dict)
            assert list(loss_weight.keys()) == ["collision", "coordinate"]
            self.loss_weight = loss_weight

    def update_data(self, state_traj, command_traj, dones_traj, coordinate_traj, init_coordinate_traj):
        """

        :param state_traj: (traj_len, n_env, state_dim)
        :param command_traj: (traj_len, n_env, command_dim)
        :param dones_traj: (traj_len, n_env)
        :param coordinate_traj: (traj_len, n_env, coordinate_dim)
        :param init_coordinate_traj: (traj_len, n_env, coordinate_dim + 1)  # include yaw
        :return: None
        """

        traj_len = state_traj.shape[0]
        n_env = state_traj.shape[1]

        new_state = np.zeros((n_env, self.state_dim))
        new_command = np.zeros((self.n_prediction_step, n_env, self.command_dim))
        new_P_col = np.zeros((self.n_prediction_step, n_env, self.P_col_dim))
        new_coordinate = np.zeros((self.n_prediction_step, n_env, self.coordinate_dim))

        for i in range(traj_len - self.n_prediction_step + 1):
            new_state = state_traj[i]

            for j in range(n_env):
                current_commands = command_traj[i:i + self.n_prediction_step, j, :]
                current_dones = dones_traj[i:i + self.n_prediction_step, j]
                current_init_coordinates = init_coordinate_traj[i, j, :]

                transition_matrix = np.array([[np.cos(current_init_coordinates[2]), np.sin(current_init_coordinates[2])],
                                              [- np.sin(current_init_coordinates[2]), np.cos(current_init_coordinates[2])]], dtype=np.float32)
                temp_coordinate_traj = coordinate_traj[i:i + self.n_prediction_step, j, :] - current_init_coordinates[:-1]
                current_coordinates = np.matmul(temp_coordinate_traj, transition_matrix.T)

                if sum(current_dones) == 0:
                    new_command[:, j, :] = current_commands
                    new_P_col[:, j, :] = current_dones[:, np.newaxis]
                    new_coordinate[:, j, :] = current_coordinates
                else:
                    done_idx = np.min(np.argwhere(current_dones == 1))
                    n_broadcast = self.n_prediction_step - (done_idx + 1)
                    P_col_broadcast = np.ones((n_broadcast, 1))  # (n_broadcast, 1)
                    command_broadcast = np.tile(current_commands[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)
                    coordinate_broadcast = np.tile(current_coordinates[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)

                    new_command[:, j, :] = np.concatenate((current_commands[:done_idx + 1], command_broadcast), axis=0)
                    new_P_col[:, j, :] = np.concatenate((current_dones[:done_idx + 1][:, np.newaxis], P_col_broadcast), axis=0)
                    new_coordinate[:, j, :] = np.concatenate((current_coordinates[:done_idx + 1], coordinate_broadcast), axis=0)

            new_state = new_state.astype(np.float32)
            new_command = new_command.astype(np.float32)
            new_P_col = new_P_col.astype(np.float32)
            new_coordinate = new_coordinate.astype(np.float32)

            # prob_col = np.sum(np.where(np.squeeze(np.sum(new_P_col, axis=0), axis=1) != 0, 1, 0)) / n_env
            # print(prob_col)

            self.storage.add_data(new_state, new_command, new_P_col, new_coordinate)

    def log(self, logging_data):
        # pass
        wandb.log(logging_data)

    def is_buffer_full(self):
        return self.storage.is_full()

    def train(self):
        mean_loss = 0
        mean_P_col_loss = 0
        mean_coordinate_loss = 0
        mean_col_prediction_accuracy = 0
        n_update = 0

        for epoch in range(self.num_learning_epochs):
            for states_batch, commands_batch, P_cols_batch, coordinates_batch \
                    in self.batch_sampler(self.mini_batch_size):
                predicted_P_cols, predicted_coordinates = self.environment_model(states_batch, commands_batch, training=True)

                traj_len = predicted_P_cols.shape[0]
                col_state = torch.where(predicted_P_cols > 0.5, 1, 0)
                col_prediction_accuracy = (torch.sum(torch.where(col_state - P_cols_batch == 0, 1, 0).squeeze(-1), dim=0) / traj_len).mean()

                # Collision probability loss (CLE)
                P_col_loss = - (P_cols_batch * torch.log(predicted_P_cols + 1e-6) + (1 - P_cols_batch) * torch.log(1 - predicted_P_cols + 1e-6))
                P_col_loss = torch.sum(P_col_loss, dim=0).mean()

                # Coordinate loss (MSE)
                coordinate_loss = torch.sum(torch.sum((predicted_coordinates - coordinates_batch).pow(2), dim=0), dim=-1).mean()

                loss = P_col_loss * self.loss_weight["collision"] + coordinate_loss * self.loss_weight["coordinate"]

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad:
                    nn.utils.clip_grad_norm_([*self.environment_model.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_loss += loss.item()
                mean_P_col_loss += P_col_loss.item()
                mean_coordinate_loss += coordinate_loss.item()
                mean_col_prediction_accuracy += col_prediction_accuracy.item()

                n_update += 1

        mean_loss /= n_update
        mean_P_col_loss /= n_update
        mean_coordinate_loss /= n_update
        mean_col_prediction_accuracy /= n_update

        self.log({"Loss/Total": mean_loss,
                  'Loss/Collision': mean_P_col_loss,
                  'Loss/Coordinate': mean_coordinate_loss,
                  'Loss/Collision_accuracy': mean_col_prediction_accuracy})

        return mean_loss, mean_P_col_loss, mean_coordinate_loss

    def evaluate(self, environment_model, state_traj, command_traj, dones_traj, coordinate_traj, init_coordinate_traj):
        """

        :param state_traj: (traj_len, n_env, state_dim)
        :param command_traj: (traj_len, n_env, command_dim)
        :param dones_traj: (traj_len, n_env)
        :param coordinate_traj: (traj_len, n_env, coordinate_dim)
        :param init_coordinate_traj: (traj_len, n_env, coordinate_dim + 1)  # include yaw
        :return:
            - real_P_cols: (n_prediction_step, n_samples, P_cols_dim)
            - real_coordinates: (n_prediction_step, n_samples, coordinate_dim)
            - predicted_P_cols: (n_prediction_step, n_samples, P_cols_dim)
            - predicted_coordinates: (n_prediction_step, n_samples, coordinate_dim)
            - mean_P_col_accuracy: double
            - mean_coordinate_error: double
        """

        traj_len = state_traj.shape[0]
        n_samples = traj_len - self.n_prediction_step + 1
        env_id = 0

        new_state = np.zeros((n_samples, self.state_dim))
        new_command = np.zeros((self.n_prediction_step, n_samples, self.command_dim))
        new_P_col = np.zeros((self.n_prediction_step, n_samples, self.P_col_dim))
        new_coordinate = np.zeros((self.n_prediction_step, n_samples, self.coordinate_dim))

        for i in range(traj_len - self.n_prediction_step + 1):
            new_state[i] = state_traj[i, env_id, :]

            current_commands = command_traj[i:i + self.n_prediction_step, env_id, :]
            current_dones = dones_traj[i:i + self.n_prediction_step, env_id]
            current_init_coordinates = init_coordinate_traj[i, env_id, :]

            transition_matrix = np.array([[np.cos(current_init_coordinates[2]), np.sin(current_init_coordinates[2])],
                                         [- np.sin(current_init_coordinates[2]), np.cos(current_init_coordinates[2])]], dtype=np.float32)
            temp_coordinate_traj = coordinate_traj[i:i + self.n_prediction_step, env_id, :] - current_init_coordinates[:-1]
            current_coordinates = np.matmul(temp_coordinate_traj, transition_matrix.T)

            if sum(current_dones) == 0:
                new_command[:, i, :] = current_commands
                new_P_col[:, i, :] = current_dones[:, np.newaxis]
                new_coordinate[:, i, :] = current_coordinates
            else:
                done_idx = np.min(np.argwhere(current_dones == 1))
                n_broadcast = self.n_prediction_step - (done_idx + 1)
                P_col_broadcast = np.ones((n_broadcast, 1))  # (n_broadcast, 1)
                command_broadcast = np.tile(current_commands[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)
                coordinate_broadcast = np.tile(current_coordinates[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)

                new_command[:, i, :] = np.concatenate((current_commands[:done_idx + 1], command_broadcast), axis=0)
                new_P_col[:, i, :] = np.concatenate((current_dones[:done_idx + 1][:, np.newaxis], P_col_broadcast), axis=0)
                new_coordinate[:, i, :] = np.concatenate((current_coordinates[:done_idx + 1], coordinate_broadcast), axis=0)

        new_state = new_state.astype(np.float32)
        new_command = new_command.astype(np.float32)
        new_P_col = new_P_col.astype(np.float32)
        new_coordinate = new_coordinate.astype(np.float32)

        real_P_cols, real_coordinates = new_P_col, new_coordinate  # ground truth
        predicted_P_cols, predicted_coordinates = environment_model(torch.from_numpy(new_state).to(self.device), torch.from_numpy(new_command).to(self.device), training=False)  # prediction

        # compute collision prediction accuracy
        predicted_col_state = np.where(predicted_P_cols > 0.5, 1, 0)
        mean_P_col_accuracy = np.mean(np.sum(predicted_col_state == real_P_cols, axis=0) / self.n_prediction_step)

        # compute coordinate prediction error
        mean_coordinate_error = np.mean(np.sum(np.sum(np.power(predicted_coordinates - real_coordinates, 2), axis=0), axis=-1))

        self.log({"Evaluate/Collistion_accuracy": mean_P_col_accuracy,
                  'Evaluate/Coordinate_error': mean_coordinate_error})

        return (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (mean_P_col_accuracy, mean_coordinate_error)










