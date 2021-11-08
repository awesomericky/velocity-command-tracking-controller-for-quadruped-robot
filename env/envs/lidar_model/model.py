import pdb

import torch.nn as nn
import numpy as np
import torch
from raisimGymTorch.algo.TCN.TCN import TemporalConvNet


class Lidar_environment_model(nn.Module):
    def __init__(self, COM_encoding_config, state_encoding_config, command_encoding_config,
                 recurrence_config, prediction_config, device):
        super(Lidar_environment_model, self).__init__()

        self.use_TCN = COM_encoding_config["use_TCN"]
        self.COM_encoding_config = COM_encoding_config
        self.state_encoding_config = state_encoding_config
        self.command_encoding_config = command_encoding_config
        self.recurrence_config = recurrence_config
        self.prediction_config = prediction_config
        self.device = device
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.prediction_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        self.set_module()

    def set_module(self):
        if self.use_TCN:
            self.COM_encoder = TemporalConvNet(num_inputs=self.COM_encoding_config["TCN"]["input"],
                                               num_channels=self.COM_encoding_config["TCN"]["output"],
                                               dropout=self.COM_encoding_config["TCN"]["dropout"],
                                               activation=self.COM_encoding_config["TCN"]["activation"],
                                               kernel_size=2,
                                               stride=1)

        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"])
        self.command_encoder = MLP(self.command_encoding_config["shape"],
                                   self.activation_map[self.command_encoding_config["activation"]],
                                   self.command_encoding_config["input"],
                                   self.command_encoding_config["output"])
        self.recurrence = torch.nn.LSTM(self.recurrence_config["input"],
                                        self.recurrence_config["hidden"],
                                        self.recurrence_config["layer"])
        self.Pcol_prediction = MLP(self.prediction_config["shape"],
                                   self.activation_map[self.prediction_config["activation"]],
                                   self.prediction_config["input"],
                                   self.prediction_config["collision"]["output"])
        self.coordinate_prediction = MLP(self.prediction_config["shape"],
                                         self.activation_map[self.prediction_config["activation"]],
                                         self.prediction_config["input"],
                                         self.prediction_config["coordinate"]["output"])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args, training=False):
        """

        :return:
            p_col: (traj_len, n_sample, 1)
            coordinate: (traj_len, n_sample, 2)
        """

        if self.use_TCN:
            """
            :param COM: (n_sample, *COM_dim)
            :param lidar: (n_sample, lidar_dim)
            :param command_traj: (traj_len, n_sample, single_command_dim)
            """
            COM, lidar, command_traj = args
            encoded_COM = self.COM_encoder(COM, only_last=True)
            state = torch.cat((encoded_COM, lidar), dim=1)
        else:
            """
            :param state: (n_sample, state_dim)
            :param command_traj: (traj_len, n_sample, single_command_dim)
            """
            state, command_traj = args

        encoded_state = self.state_encoder.architecture(state).unsqueeze(0)
        traj_len, n_sample, single_command_dim = command_traj.shape
        command_traj = command_traj.reshape(-1, single_command_dim)
        encoded_command = self.command_encoder.architecture(command_traj).reshape(traj_len, n_sample, -1)
        encoded_prediction, (_, _) = self.recurrence(encoded_command, (encoded_state, encoded_state))
        traj_len, n_sample, encoded_prediction_dim = encoded_prediction.shape
        encoded_prediction = encoded_prediction.reshape(-1, encoded_prediction_dim)
        collision_prob_traj = self.sigmoid(self.Pcol_prediction.architecture(encoded_prediction))
        collision_prob_traj = collision_prob_traj.reshape(traj_len, n_sample, self.prediction_config["collision"]["output"])
         
        coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        coordinate_traj = coordinate_traj.reshape(traj_len, n_sample, self.prediction_config["coordinate"]["output"])
        """
        delata_coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        delata_coordinate_traj = delata_coordinate_traj.reshape(traj_len, n_sample, self.prediction_config["coordinate"]["output"])

        coordinate_traj = torch.zeros(traj_len, n_sample, self.prediction_config["coordinate"]["output"]).to(self.device)
        for i in range(traj_len):
            if i == 0:
                coordinate_traj[i, :, :] = delata_coordinate_traj[i, :, :]
            else:
                coordinate_traj[i, :, :] = coordinate_traj[i - 1, :, :] + delata_coordinate_traj[i, :, :]
        """
        if training:
            # return "device" torch tensor
            return collision_prob_traj, coordinate_traj
        else:
            # return "cpu" numpy tensor
            return collision_prob_traj.cpu().detach().numpy(), coordinate_traj.cpu().detach().numpy()


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, dropout=0.0, batchnorm=False):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            if batchnorm:
                modules.append(nn.BatchNorm1d(shape[idx+1]))
            modules.append(self.activation_fn())
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
