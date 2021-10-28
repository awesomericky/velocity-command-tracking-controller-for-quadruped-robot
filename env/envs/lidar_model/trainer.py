from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .storage import DataStorage, DataStorage_TCN, Buffer
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
                 max_grad_norm=0.5,
                 logging=True,
                 P_col_interpolate=False,
                 prioritized_data_update=True,
                 prioritized_data_update_magnitude=0.5):

        self.n_prediction_step = int(prediction_period / delta_prediction_time)
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.P_col_dim = P_col_dim
        self.coordinate_dim = coordinate_dim

        self.prioritized_data_update = prioritized_data_update
        self.prioritized_data_update_magnitude = prioritized_data_update_magnitude

        # environment model
        self.environment_model = environment_model
        self.environment_model.train()
        self.environment_model.to(device)

        self.storage = DataStorage(max_storage_size=max_storage_size,
                                   state_dim=state_dim,
                                   command_shape=[self.n_prediction_step, command_dim],
                                   P_col_shape=[self.n_prediction_step, P_col_dim],
                                   coordinate_shape=[self.n_prediction_step, coordinate_dim],
                                   device=device,
                                   prioritized_data_update=self.prioritized_data_update)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.environment_model.parameters()], lr=learning_rate)
        self.device = device
        self.logging = logging
        self.P_col_interpolate = P_col_interpolate

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

        if self.prioritized_data_update:
            total_new_state = []
            total_new_command = []
            total_new_P_col = []
            total_new_corrdinate = []

        n_traj_step_samples = int((traj_len - self.n_prediction_step + 1) / 2)
        sampled_traj_steps = np.random.choice(traj_len - self.n_prediction_step + 1, n_traj_step_samples, replace=False)

        for i in sampled_traj_steps:
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
                    if self.P_col_interpolate:
                        interpolate_P_col = np.linspace(0., 1., done_idx + 2)[1:]
                        new_P_col[:, j, :] = np.concatenate((interpolate_P_col[:, np.newaxis], P_col_broadcast), axis=0)
                    else:
                        new_P_col[:, j, :] = np.concatenate((current_dones[:done_idx + 1][:, np.newaxis], P_col_broadcast), axis=0)
                    new_coordinate[:, j, :] = np.concatenate((current_coordinates[:done_idx + 1], coordinate_broadcast), axis=0)

            new_state = new_state.astype(np.float32)
            new_command = new_command.astype(np.float32)
            new_P_col = new_P_col.astype(np.float32)
            new_coordinate = new_coordinate.astype(np.float32)

            # prob_col = np.sum(np.where(np.squeeze(np.sum(new_P_col, axis=0), axis=1) != 0, 1, 0)) / n_env
            # print(prob_col)

            if self.prioritized_data_update and self.storage.is_full():
                total_new_state.append(torch.from_numpy(new_state.copy()).to(self.device))
                total_new_command.append(torch.from_numpy(new_command.copy()).to(self.device))
                total_new_P_col.append(torch.from_numpy(new_P_col.copy()).to(self.device))
                total_new_corrdinate.append(torch.from_numpy(new_coordinate.copy()).to(self.device))
            else:
                self.storage.add_data(new_state, new_command, new_P_col, new_coordinate)

        if self.prioritized_data_update and self.storage.is_full():
            buffer_states, buffer_commands, buffer_P_cols, buffer_coordinates = self.storage.return_data()
            total_new_state.append(buffer_states)
            total_new_command.append(buffer_commands)
            total_new_P_col.append(buffer_P_cols)
            total_new_corrdinate.append(buffer_coordinates)

            total_new_state = torch.cat(total_new_state, dim=0)
            total_new_command = torch.cat(total_new_command, dim=1)
            total_new_P_col = torch.cat(total_new_P_col, dim=1)
            total_new_corrdinate = torch.cat(total_new_corrdinate, dim=1)

            with torch.no_grad():
                predicted_P_cols, predicted_coordinates = self.environment_model(total_new_state, total_new_command, training=True)
                # Collision probability loss (CLE)
                P_col_loss = - (total_new_P_col * torch.log(predicted_P_cols + 1e-6) + (1 - total_new_P_col) * torch.log(1 - predicted_P_cols + 1e-6))
                P_col_loss = torch.sum(P_col_loss, dim=0).squeeze(-1)

                # Coordinate loss (MSE)
                coordinate_loss = torch.sum(torch.sum((predicted_coordinates - total_new_corrdinate).pow(2), dim=0), dim=-1)

                loss = P_col_loss * self.loss_weight["collision"] + coordinate_loss * self.loss_weight["coordinate"]
                PER_prob = loss ** self.prioritized_data_update_magnitude
                PER_prob /= torch.sum(PER_prob)

                PER_sampled_idx = torch.multinomial(PER_prob, self.max_storage_size, replacement=False)
                new_buffer_states = total_new_state[PER_sampled_idx, :]
                new_buffer_commands = total_new_command[:, PER_sampled_idx, :]
                new_buffer_P_cols = total_new_P_col[:, PER_sampled_idx, :]
                new_buffer_coordinates = total_new_corrdinate[:, PER_sampled_idx, :]
                self.storage.update_buffer_data(new_buffer_states, new_buffer_commands, new_buffer_P_cols, new_buffer_coordinates)

    def log(self, logging_data):
        if self.logging:
            wandb.log(logging_data)
        else:
            pass

    def is_buffer_full(self):
        return self.storage.is_full()

    def train(self):
        mean_loss = 0
        mean_P_col_loss = 0
        mean_coordinate_loss = 0
        mean_col_prediction_accuracy = 0
        mean_not_col_prediction_accuracy = 0
        n_update = 0
        n_col_prediction_acc_mean = 0
        n_not_col_prediction_acc_mean = 0

        for epoch in range(self.num_learning_epochs):
            for states_batch, commands_batch, P_cols_batch, coordinates_batch \
                    in self.batch_sampler(self.mini_batch_size):
                predicted_P_cols, predicted_coordinates = self.environment_model(states_batch, commands_batch, training=True)

                traj_len = predicted_P_cols.shape[0]
                if self.P_col_interpolate:
                    col_state = torch.where(predicted_P_cols > 0.99, 1, 0)
                    ground_truth_col_state = torch.where(P_cols_batch > 0.99, 1, 0)
                    n_total_col = torch.sum(ground_truth_col_state)
                    n_total_not_col = torch.sum(1 - ground_truth_col_state)
                    if n_total_col != 0:
                        col_prediction_accuracy = torch.sum(torch.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
                    if n_total_not_col != 0:
                        not_col_prediction_accuracy = torch.sum(torch.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col
                else:
                    col_state = torch.where(predicted_P_cols > 0.5, 1, 0)
                    ground_truth_col_state = torch.where(P_cols_batch > 0.5, 1, 0)
                    n_total_col = torch.sum(ground_truth_col_state)
                    n_total_not_col = torch.sum(1 - ground_truth_col_state)
                    if n_total_col != 0:
                        col_prediction_accuracy = torch.sum(torch.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
                    if n_total_not_col != 0:
                        not_col_prediction_accuracy = torch.sum(torch.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col

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
                if n_total_col != 0:
                    mean_col_prediction_accuracy += col_prediction_accuracy.item()
                    n_col_prediction_acc_mean += 1
                if n_total_not_col != 0:
                    mean_not_col_prediction_accuracy += not_col_prediction_accuracy.item()
                    n_not_col_prediction_acc_mean += 1

                n_update += 1
        
        mean_loss /= n_update
        mean_P_col_loss /= n_update
        mean_coordinate_loss /= n_update

        logging_data = dict()
        logging_data['Loss/Total'] = mean_loss
        logging_data['Loss/Collision'] = mean_P_col_loss
        logging_data['Loss/Coordinate'] = mean_coordinate_loss

        if n_col_prediction_acc_mean != 0:
            mean_col_prediction_accuracy /= n_col_prediction_acc_mean
            logging_data['Loss/Collision_accuracy'] = mean_col_prediction_accuracy
        else:
            mean_col_prediction_accuracy = 0

        if n_not_col_prediction_acc_mean != 0:
            mean_not_col_prediction_accuracy /= n_not_col_prediction_acc_mean
            logging_data['Loss/No_Collision_accuracy'] = mean_not_col_prediction_accuracy
        else:
            mean_not_col_prediction_accuracy = 0

        self.log(logging_data)

        return mean_loss, mean_P_col_loss, mean_coordinate_loss, mean_col_prediction_accuracy, mean_not_col_prediction_accuracy

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
        n_env = state_traj.shape[1]
        n_samples = (traj_len - self.n_prediction_step + 1) * n_env

        new_state = np.zeros((n_samples, self.state_dim))
        new_command = np.zeros((self.n_prediction_step, n_samples, self.command_dim))
        new_P_col = np.zeros((self.n_prediction_step, n_samples, self.P_col_dim))
        new_coordinate = np.zeros((self.n_prediction_step, n_samples, self.coordinate_dim))

        for i in range(traj_len - self.n_prediction_step + 1):
            new_state[i*n_env : (i+1)*n_env] = state_traj[i]

            for env_id in range(n_env):
                current_commands = command_traj[i:i + self.n_prediction_step, env_id, :]
                current_dones = dones_traj[i:i + self.n_prediction_step, env_id]
                current_init_coordinates = init_coordinate_traj[i, env_id, :]

                transition_matrix = np.array([[np.cos(current_init_coordinates[2]), np.sin(current_init_coordinates[2])],
                                             [- np.sin(current_init_coordinates[2]), np.cos(current_init_coordinates[2])]], dtype=np.float32)
                temp_coordinate_traj = coordinate_traj[i:i + self.n_prediction_step, env_id, :] - current_init_coordinates[:-1]
                current_coordinates = np.matmul(temp_coordinate_traj, transition_matrix.T)

                if sum(current_dones) == 0:
                    new_command[:, i*n_env + env_id, :] = current_commands
                    new_P_col[:, i*n_env + env_id, :] = current_dones[:, np.newaxis]
                    new_coordinate[:, i*n_env + env_id, :] = current_coordinates
                else:
                    done_idx = np.min(np.argwhere(current_dones == 1))
                    n_broadcast = self.n_prediction_step - (done_idx + 1)
                    P_col_broadcast = np.ones((n_broadcast, 1))  # (n_broadcast, 1)
                    command_broadcast = np.tile(current_commands[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)
                    coordinate_broadcast = np.tile(current_coordinates[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)

                    new_command[:, i*n_env + env_id, :] = np.concatenate((current_commands[:done_idx + 1], command_broadcast), axis=0)
                    if self.P_col_interpolate:
                        interpolate_P_col = np.linspace(0., 1., done_idx + 2)[1:]
                        new_P_col[:, i*n_env + env_id, :] = np.concatenate((interpolate_P_col[:, np.newaxis], P_col_broadcast), axis=0)
                    else:
                        new_P_col[:, i*n_env + env_id, :] = np.concatenate((current_dones[:done_idx + 1][:, np.newaxis], P_col_broadcast), axis=0)
                    new_coordinate[:, i*n_env + env_id, :] = np.concatenate((current_coordinates[:done_idx + 1], coordinate_broadcast), axis=0)

        new_state = new_state.astype(np.float32)
        new_command = new_command.astype(np.float32)
        new_P_col = new_P_col.astype(np.float32)
        new_coordinate = new_coordinate.astype(np.float32)

        real_P_cols, real_coordinates = new_P_col, new_coordinate  # ground truth
        predicted_P_cols, predicted_coordinates = environment_model(torch.from_numpy(new_state).to(self.device),
                                                                    torch.from_numpy(new_command).to(self.device),
                                                                    training=False)  # prediction

        # compute collision prediction accuracy
        if self.P_col_interpolate:
            col_state = np.where(predicted_P_cols > 0.99, 1, 0)
            ground_truth_col_state = np.where(real_P_cols == 1., 1, 0)
            n_total_col = np.sum(ground_truth_col_state)
            n_total_not_col = np.sum(1 - ground_truth_col_state)
            if n_total_col != 0:
                col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
            else:
                col_prediction_accuracy = -1
            if n_total_not_col != 0:
                not_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col
            else:
                not_col_prediction_accuracy = -1
            coll_correct_idx = np.where(col_state + ground_truth_col_state == 2, 1, 0)
            not_coll_correct_idx = np.where(col_state + ground_truth_col_state == 0, 1, 0)
            total_col_prediction_accuracy = np.sum(coll_correct_idx + not_coll_correct_idx) / (n_total_col + n_total_not_col)
        else:
            col_state = np.where(predicted_P_cols > 0.5, 1, 0)
            ground_truth_col_state = np.where(real_P_cols == 1., 1, 0)
            n_total_col = np.sum(ground_truth_col_state)
            n_total_not_col = np.sum(1 - ground_truth_col_state)
            if n_total_col != 0:
                col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
            else:
                col_prediction_accuracy = -1
            if n_total_not_col != 0:
                not_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col
            else:
                not_col_prediction_accuracy = -1
            coll_correct_idx = np.where(col_state + ground_truth_col_state == 2, 1, 0)
            not_coll_correct_idx = np.where(col_state + ground_truth_col_state == 0, 1, 0)
            total_col_prediction_accuracy = np.sum(coll_correct_idx + not_coll_correct_idx) / (n_total_col + n_total_not_col)

        # compute coordinate prediction error
        # mean of coordinate error distance sum for each trajectory
        # (divide the value by number of prediction step to compute the average coordinate distance error for each step)
        mean_coordinate_error = np.mean(np.sum(np.sqrt(np.sum(np.power(predicted_coordinates - real_coordinates, 2), axis=-1)), axis=0))

        logging_data = dict()
        if n_total_col != 0:
            logging_data['Evaluate/Collistion_accuracy'] = col_prediction_accuracy
        if n_total_not_col != 0:
            logging_data['Evaluate/No_collistion_accuracy'] = not_col_prediction_accuracy
        logging_data['Evaluate/Total_collistion_accuracy'] = total_col_prediction_accuracy
        logging_data['Evaluate/Coordinate_error'] = mean_coordinate_error
        self.log(logging_data)

        return (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (total_col_prediction_accuracy, col_prediction_accuracy, not_col_prediction_accuracy, mean_coordinate_error)

    # def evaluate(self, environment_model, state_traj, command_traj, dones_traj, coordinate_traj, init_coordinate_traj):
    #     """
    #
    #     :param state_traj: (traj_len, n_env, state_dim)
    #     :param command_traj: (traj_len, n_env, command_dim)
    #     :param dones_traj: (traj_len, n_env)
    #     :param coordinate_traj: (traj_len, n_env, coordinate_dim)
    #     :param init_coordinate_traj: (traj_len, n_env, coordinate_dim + 1)  # include yaw
    #     :return:
    #         - real_P_cols: (n_prediction_step, n_samples, P_cols_dim)
    #         - real_coordinates: (n_prediction_step, n_samples, coordinate_dim)
    #         - predicted_P_cols: (n_prediction_step, n_samples, P_cols_dim)
    #         - predicted_coordinates: (n_prediction_step, n_samples, coordinate_dim)
    #         - mean_P_col_accuracy: double
    #         - mean_coordinate_error: double
    #     """
    #
    #     traj_len = state_traj.shape[0]
    #     n_samples = traj_len - self.n_prediction_step + 1
    #     env_id = 0
    #
    #     new_state = np.zeros((n_samples, self.state_dim))
    #     new_command = np.zeros((self.n_prediction_step, n_samples, self.command_dim))
    #     new_P_col = np.zeros((self.n_prediction_step, n_samples, self.P_col_dim))
    #     new_coordinate = np.zeros((self.n_prediction_step, n_samples, self.coordinate_dim))
    #
    #     for i in range(traj_len - self.n_prediction_step + 1):
    #         new_state[i] = state_traj[i, env_id, :]
    #
    #         current_commands = command_traj[i:i + self.n_prediction_step, env_id, :]
    #         current_dones = dones_traj[i:i + self.n_prediction_step, env_id]
    #         current_init_coordinates = init_coordinate_traj[i, env_id, :]
    #
    #         transition_matrix = np.array([[np.cos(current_init_coordinates[2]), np.sin(current_init_coordinates[2])],
    #                                       [- np.sin(current_init_coordinates[2]), np.cos(current_init_coordinates[2])]], dtype=np.float32)
    #         temp_coordinate_traj = coordinate_traj[i:i + self.n_prediction_step, env_id, :] - current_init_coordinates[:-1]
    #         current_coordinates = np.matmul(temp_coordinate_traj, transition_matrix.T)
    #
    #         if sum(current_dones) == 0:
    #             new_command[:, i, :] = current_commands
    #             new_P_col[:, i, :] = current_dones[:, np.newaxis]
    #             new_coordinate[:, i, :] = current_coordinates
    #         else:
    #             done_idx = np.min(np.argwhere(current_dones == 1))
    #             n_broadcast = self.n_prediction_step - (done_idx + 1)
    #             P_col_broadcast = np.ones((n_broadcast, 1))  # (n_broadcast, 1)
    #             command_broadcast = np.tile(current_commands[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)
    #             coordinate_broadcast = np.tile(current_coordinates[done_idx], (n_broadcast, 1))  # (n_broadcast, 3)
    #
    #             new_command[:, i, :] = np.concatenate((current_commands[:done_idx + 1], command_broadcast), axis=0)
    #             if self.P_col_interpolate:
    #                 interpolate_P_col = np.linspace(0., 1., done_idx + 2)[1:]
    #                 new_P_col[:, i, :] = np.concatenate((interpolate_P_col[:, np.newaxis], P_col_broadcast), axis=0)
    #             else:
    #                 new_P_col[:, i, :] = np.concatenate((current_dones[:done_idx + 1][:, np.newaxis], P_col_broadcast), axis=0)
    #             new_coordinate[:, i, :] = np.concatenate((current_coordinates[:done_idx + 1], coordinate_broadcast), axis=0)
    #
    #     new_state = new_state.astype(np.float32)
    #     new_command = new_command.astype(np.float32)
    #     new_P_col = new_P_col.astype(np.float32)
    #     new_coordinate = new_coordinate.astype(np.float32)
    #
    #     real_P_cols, real_coordinates = new_P_col, new_coordinate  # ground truth
    #     predicted_P_cols, predicted_coordinates = environment_model(torch.from_numpy(new_state).to(self.device),
    #                                                                 torch.from_numpy(new_command).to(self.device),
    #                                                                 training=False)  # prediction
    #
    #     # compute collision prediction accuracy
    #     if self.P_col_interpolate:
    #         col_state = np.where(predicted_P_cols > 0.99, 1, 0)
    #         ground_truth_col_state = np.where(real_P_cols == 1., 1, 0)
    #         n_total_col = np.sum(ground_truth_col_state)
    #         n_total_not_col = np.sum(1 - ground_truth_col_state)
    #         if n_total_col != 0:
    #             col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
    #         else:
    #             col_prediction_accuracy = -1
    #         if n_total_not_col != 0:
    #             not_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col
    #         else:
    #             not_col_prediction_accuracy = -1
    #         pdb.set_trace()
    #         total_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0 or col_state + ground_truth_col_state == 2, 1, 0)) / (n_total_col + n_total_not_col)
    #     else:
    #         col_state = np.where(predicted_P_cols > 0.5, 1, 0)
    #         ground_truth_col_state = np.where(real_P_cols == 1., 1, 0)
    #         n_total_col = np.sum(ground_truth_col_state)
    #         n_total_not_col = np.sum(1 - ground_truth_col_state)
    #         if n_total_col != 0:
    #             col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 2, 1, 0)) / n_total_col
    #         else:
    #             col_prediction_accuracy = -1
    #         if n_total_not_col != 0:
    #             not_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0, 1, 0)) / n_total_not_col
    #         else:
    #             not_col_prediction_accuracy = -1
    #         total_col_prediction_accuracy = np.sum(np.where(col_state + ground_truth_col_state == 0 or col_state + ground_truth_col_state == 2, 1, 0)) / (n_total_col + n_total_not_col)
    #
    #     # compute coordinate prediction error
    #     mean_coordinate_error = np.mean(np.sum(np.sum(np.power(predicted_coordinates - real_coordinates, 2), axis=0), axis=-1))
    #
    #     logging_data = dict()
    #     if n_total_col != 0:
    #         logging_data['Evaluate/Collistion_accuracy'] = col_prediction_accuracy
    #     if n_total_not_col != 0:
    #         logging_data['Evaluate/No_collistion_accuracy'] = not_col_prediction_accuracy
    #     logging_data['Evaluate/Total_collistion_accuracy'] = total_col_prediction_accuracy
    #     logging_data['Evaluate/Coordinate_error'] = mean_coordinate_error
    #     self.log(logging_data)
    #
    #     return (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (total_col_prediction_accuracy, col_prediction_accuracy, not_col_prediction_accuracy, mean_coordinate_error)


class Trainer_TCN:
    def __init__(self,
                 environment_model,
                 COM_dim,
                 lidar_dim,
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
                 max_grad_norm=0.5,
                 logging=True):

        self.n_prediction_step = int(prediction_period / delta_prediction_time)
        self.COM_dim = COM_dim
        self.lidar_dim = lidar_dim
        self.command_dim = command_dim
        self.P_col_dim = P_col_dim
        self.coordinate_dim = coordinate_dim

        # environment model
        self.environment_model = environment_model
        self.environment_model.train()
        self.environment_model.to(device)

        self.storage = DataStorage_TCN(max_storage_size=max_storage_size,
                                       COM_dim=COM_dim,
                                       lidar_dim=lidar_dim,
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
        self.logging = logging

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

    def update_data(self, COM_traj, lidar_traj, command_traj, dones_traj, coordinate_traj, init_coordinate_traj):
        """

        :param COM_traj: (traj_len, n_env, *COM_dim)
        :param lidar_traj: (traj_len, n_env, lidar_dim)
        :param command_traj: (traj_len, n_env, command_dim)
        :param dones_traj: (traj_len, n_env)
        :param coordinate_traj: (traj_len, n_env, coordinate_dim)
        :param init_coordinate_traj: (traj_len, n_env, coordinate_dim + 1)  # include yaw
        :return: None
        """

        traj_len = COM_traj.shape[0]
        n_env = COM_traj.shape[1]

        new_COM = np.zeros((n_env, *self.COM_dim))
        new_lidar = np.zeros((n_env, self.lidar_dim))
        new_command = np.zeros((self.n_prediction_step, n_env, self.command_dim))
        new_P_col = np.zeros((self.n_prediction_step, n_env, self.P_col_dim))
        new_coordinate = np.zeros((self.n_prediction_step, n_env, self.coordinate_dim))

        for i in range(traj_len - self.n_prediction_step + 1):
            new_COM = COM_traj[i]
            new_lidar = lidar_traj[i]

            for j in range(n_env):
                current_commands = command_traj[i:i + self.n_prediction_step, j, :]
                current_dones = dones_traj[i:i + self.n_prediction_step, j]
                current_init_coordinates = init_coordinate_traj[i, j, :]

                # WORLD frame --> LOCAL frame
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

            new_COM = new_COM.astype(np.float32)
            new_lidar = new_lidar.astype(np.float32)
            new_command = new_command.astype(np.float32)
            new_P_col = new_P_col.astype(np.float32)
            new_coordinate = new_coordinate.astype(np.float32)

            # prob_col = np.sum(np.where(np.squeeze(np.sum(new_P_col, axis=0), axis=1) != 0, 1, 0)) / n_env
            # print(prob_col)

            self.storage.add_data(new_COM, new_lidar, new_command, new_P_col, new_coordinate)

    def log(self, logging_data):
        if self.logging:
            wandb.log(logging_data)
        else:
            pass

    def is_buffer_full(self):
        return self.storage.is_full()

    def train(self):
        mean_loss = 0
        mean_P_col_loss = 0
        mean_coordinate_loss = 0
        mean_col_prediction_accuracy = 0
        n_update = 0

        for epoch in range(self.num_learning_epochs):
            for COMs_batch, lidars_batch, commands_batch, P_cols_batch, coordinates_batch \
                    in self.batch_sampler(self.mini_batch_size):

                # Check the percentage of collision trajectory in the dataset  # 35%
                # print(len(np.where(np.sum(np.where(np.squeeze(P_cols_batch.cpu().numpy(), -1) > 0.5, 1, 0), 0) != 0)[0]) / COMs_batch.shape[0])

                predicted_P_cols, predicted_coordinates = self.environment_model(COMs_batch, lidars_batch, commands_batch, training=True)

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

    def evaluate(self, environment_model, COM_traj, lidar_traj, command_traj, dones_traj, coordinate_traj, init_coordinate_traj):
        """

        :param COM_traj: (traj_len, n_env, *COM_dim)
        :param lidar_traj: (traj_len, n_env, lidar_dim)
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

        traj_len = COM_traj.shape[0]
        n_samples = traj_len - self.n_prediction_step + 1
        env_id = 0

        new_COM = np.zeros((n_samples, *self.COM_dim))
        new_lidar = np.zeros((n_samples, self.lidar_dim))
        new_command = np.zeros((self.n_prediction_step, n_samples, self.command_dim))
        new_P_col = np.zeros((self.n_prediction_step, n_samples, self.P_col_dim))
        new_coordinate = np.zeros((self.n_prediction_step, n_samples, self.coordinate_dim))

        for i in range(traj_len - self.n_prediction_step + 1):
            new_COM[i] = COM_traj[i, env_id, :, :]
            new_lidar[i] = lidar_traj[i, env_id, :]

            current_commands = command_traj[i:i + self.n_prediction_step, env_id, :]
            current_dones = dones_traj[i:i + self.n_prediction_step, env_id]
            current_init_coordinates = init_coordinate_traj[i, env_id, :]

            # WORLD frame --> LOCAL frame
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

        new_COM = new_COM.astype(np.float32)
        new_lidar = new_lidar.astype(np.float32)
        new_command = new_command.astype(np.float32)
        new_P_col = new_P_col.astype(np.float32)
        new_coordinate = new_coordinate.astype(np.float32)

        real_P_cols, real_coordinates = new_P_col, new_coordinate  # ground truth
        predicted_P_cols, predicted_coordinates = environment_model(torch.from_numpy(new_COM).to(self.device),
                                                                    torch.from_numpy(new_lidar).to(self.device),
                                                                    torch.from_numpy(new_command).to(self.device),
                                                                    training=False)  # prediction

        # compute collision prediction accuracy
        predicted_col_state = np.where(predicted_P_cols > 0.5, 1, 0)
        mean_P_col_accuracy = np.mean(np.sum(predicted_col_state == real_P_cols, axis=0) / self.n_prediction_step)

        # compute coordinate prediction error
        mean_coordinate_error = np.mean(np.sum(np.sum(np.power(predicted_coordinates - real_coordinates, 2), axis=0), axis=-1))

        self.log({"Evaluate/Collistion_accuracy": mean_P_col_accuracy,
                  'Evaluate/Coordinate_error': mean_coordinate_error})

        return (real_P_cols, real_coordinates), (predicted_P_cols, predicted_coordinates), (mean_P_col_accuracy, mean_coordinate_error)










