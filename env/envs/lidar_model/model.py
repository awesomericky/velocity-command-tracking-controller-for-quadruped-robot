import pdb

import torch.nn as nn
import numpy as np
import torch
from raisimGymTorch.algo.TCN.TCN import TemporalConvNet
import pdb


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
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.command_encoder = MLP(self.command_encoding_config["shape"],
                                   self.activation_map[self.command_encoding_config["activation"]],
                                   self.command_encoding_config["input"],
                                   self.command_encoding_config["output"],
                                   dropout=self.command_encoding_config["dropout"],
                                   batchnorm=self.command_encoding_config["batchnorm"])
        self.recurrence = torch.nn.LSTM(self.recurrence_config["input"],
                                        self.recurrence_config["hidden"],
                                        self.recurrence_config["layer"])
        self.Pcol_prediction = MLP(self.prediction_config["shape"],
                                   self.activation_map[self.prediction_config["activation"]],
                                   self.prediction_config["input"],
                                   self.prediction_config["collision"]["output"],
                                   dropout=self.prediction_config["dropout"],
                                   batchnorm=self.prediction_config["batchnorm"])
        self.coordinate_prediction = MLP(self.prediction_config["shape"],
                                         self.activation_map[self.prediction_config["activation"]],
                                         self.prediction_config["input"],
                                         self.prediction_config["coordinate"]["output"],
                                         dropout=self.prediction_config["dropout"],
                                         batchnorm=self.prediction_config["batchnorm"])
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
         
        # coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        # coordinate_traj = coordinate_traj.reshape(traj_len, n_sample, self.prediction_config["coordinate"]["output"])

        delata_coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        delata_coordinate_traj = delata_coordinate_traj.reshape(traj_len, n_sample, self.prediction_config["coordinate"]["output"])

        coordinate_traj = torch.zeros(traj_len, n_sample, self.prediction_config["coordinate"]["output"]).to(self.device)
        for i in range(traj_len):
            if i == 0:
                coordinate_traj[i, :, :] = delata_coordinate_traj[i, :, :]
            else:
                coordinate_traj[i, :, :] = coordinate_traj[i - 1, :, :] + delata_coordinate_traj[i, :, :]

        if training:
            # return "device" torch tensor
            return collision_prob_traj, coordinate_traj
        else:
            # return "cpu" numpy tensor
            return collision_prob_traj.cpu().detach().numpy(), coordinate_traj.cpu().detach().numpy()


class CVAE_implicit_distribution(nn.Module):
    def __init__(self,
                 state_encoding_config,
                 command_encoding_config,
                 recurrence_encoding_config,
                 latent_encoding_config,
                 latent_decoding_config,
                 recurrence_decoding_config,
                 command_decoding_config,
                 device,
                 pretrained_weight,
                 state_encoder_fixed=True,
                 command_encoder_fixed=True):

        super(CVAE_implicit_distribution, self).__init__()

        self.state_encoding_config = state_encoding_config
        self.command_encoding_config = command_encoding_config
        self.recurrence_encoding_config = recurrence_encoding_config
        self.latent_encoding_config = latent_encoding_config
        self.latent_decoding_config = latent_decoding_config
        self.recurrence_decoding_config = recurrence_decoding_config
        self.command_decoding_config = command_decoding_config
        self.device = device
        self.pretrained_weight = pretrained_weight
        self.state_encoder_fixed = state_encoder_fixed
        self.command_encoder_fixed = command_encoder_fixed
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}

        assert self.state_encoder_fixed, "State encoder is recommanded to be fixed"
        assert self.command_encoder_fixed, "Command encoder is recommanded to be fixed"

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.recurrence_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.recurrence_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        self.set_module()

    def set_module(self):
        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.command_encoder = MLP(self.command_encoding_config["shape"],
                                   self.activation_map[self.command_encoding_config["activation"]],
                                   self.command_encoding_config["input"],
                                   self.command_encoding_config["output"],
                                   dropout=self.state_encoding_config["dropout"],
                                   batchnorm=self.state_encoding_config["batchnorm"])
        self.recurrence_encoder = torch.nn.GRU(self.recurrence_encoding_config["input"],
                                                self.recurrence_encoding_config["hidden"],
                                                self.recurrence_encoding_config["layer"])
        self.latent_mean_encoder = MLP(self.latent_encoding_config["shape"],
                                       self.activation_map[self.latent_encoding_config["activation"]],
                                       self.latent_encoding_config["input"],
                                       self.latent_encoding_config["output"],
                                       dropout=self.state_encoding_config["dropout"],
                                       batchnorm=self.state_encoding_config["batchnorm"])
        self.latent_log_var_encoder = MLP(self.latent_encoding_config["shape"],
                                          self.activation_map[self.latent_encoding_config["activation"]],
                                          self.latent_encoding_config["input"],
                                          self.latent_encoding_config["output"],
                                          dropout=self.state_encoding_config["dropout"],
                                          batchnorm=self.state_encoding_config["batchnorm"])
        self.latent_decoder = MLP(self.latent_decoding_config["shape"],
                                  self.activation_map[self.latent_decoding_config["activation"]],
                                  self.latent_decoding_config["input"],
                                  self.latent_decoding_config["output"],
                                  dropout=self.state_encoding_config["dropout"],
                                  batchnorm=self.state_encoding_config["batchnorm"])
        self.recurrence_decoder = torch.nn.GRU(self.recurrence_decoding_config["input"],
                                               self.recurrence_decoding_config["hidden"],
                                               self.recurrence_decoding_config["layer"])
        self.command_decoder = MLP(self.command_decoding_config["shape"],
                                   self.activation_map[self.command_decoding_config["activation"]],
                                   self.command_decoding_config["input"],
                                   self.command_decoding_config["output"],
                                   dropout=self.state_encoding_config["dropout"],
                                   batchnorm=self.state_encoding_config["batchnorm"])

        if self.state_encoder_fixed:
            # load pretrained state encoder
            state_encoder_state_dict = self.state_encoder.state_dict()
            pretrained_state_encoder_state_dict = {k: v for k, v in self.pretrained_weight.items() if k in state_encoder_state_dict}
            state_encoder_state_dict.update(pretrained_state_encoder_state_dict)
            self.state_encoder.load_state_dict(state_encoder_state_dict)
            self.state_encoder.eval()

        if self.command_encoder_fixed:
            # load pretrained state encoder
            command_encoder_state_dict = self.command_encoder.state_dict()
            pretrained_command_encoder_state_dict = {k: v for k, v in self.pretrained_weight.items() if k in command_encoder_state_dict}
            command_encoder_state_dict.update(pretrained_command_encoder_state_dict)
            self.command_encoder.load_state_dict(command_encoder_state_dict)
            self.command_encoder.eval()

    def forward(self, *args, training=False):
        """

            :param state: (n_sample, state_dim)
            :param command_traj: (traj_len, n_sample, single_command_dim)

            :return:
            latent_mean: (n_sample, latent_dim)
            latent_log_var: (n_sample, latent_dim)
            sampled_command_traj: (traj_len, n_sample, 3)
        """
        state, command_traj = args

        # state encoding
        if self.state_encoder_fixed:
            with torch.no_grad():
                encoded_state = self.state_encoder.architecture(state)
        else:
            encoded_state = self.state_encoder.architecture(state)

        # command encoding
        traj_len, n_sample, single_command_dim = command_traj.shape
        command_traj = command_traj.reshape(-1, single_command_dim)
        if self.command_encoder_fixed:
            with torch.no_grad():
                encoded_command = self.command_encoder.architecture(command_traj).reshape(traj_len, n_sample, -1)
        else:
            encoded_command = self.command_encoder.architecture(command_traj).reshape(traj_len, n_sample, -1)

        # command trajectory encoding
        _, encoded_command_traj = self.recurrence_encoder(encoded_command)
        encoded_command_traj = encoded_command_traj.squeeze(0)

        # predict posterior distribution in latent space
        total_encoded_result = torch.cat((encoded_state, encoded_command_traj), dim=1)  # (n_sample, encoded_dim)
        latent_mean = self.latent_mean_encoder.architecture(total_encoded_result)
        latent_log_var = self.latent_log_var_encoder.architecture(total_encoded_result)

        # sample with reparameterization trick
        latent_std = torch.exp(0.5 * latent_log_var)
        eps = torch.rand_like(latent_std)
        sample = latent_mean + (eps * latent_std)

        # decode command trajectory
        total_decoded_result = torch.cat((encoded_state, sample), dim=1)
        hidden_state = self.latent_decoder.architecture(total_decoded_result).unsqueeze(0)
        decoded_traj = torch.zeros(traj_len, n_sample, self.recurrence_decoding_config["hidden"]).to(self.device)
        input = torch.zeros(1, n_sample, self.recurrence_decoding_config["input"]).to(self.device)

        for i in range(traj_len):
            output, hidden_state = self.recurrence_decoder(input, hidden_state)
            output = output.squeeze(0)
            decoded_traj[i] = output
            input = decoded_traj[i]

        decoded_traj = decoded_traj.reshape(-1, self.recurrence_decoding_config["hidden"])
        sampled_command_traj = self.command_decoder.architecture(decoded_traj)
        sampled_command_traj = sampled_command_traj.reshape(traj_len, n_sample, -1)

        if training:
            return latent_mean, latent_log_var, sampled_command_traj
        else:
            return latent_mean.cpu().detach().numpy(), latent_log_var.cpu().detach().numpy(), sampled_command_traj.cpu().detach().numpy()


class CVAE_implicit_distribution_inference(nn.Module):
    def __init__(self,
                 state_encoding_config,
                 latent_decoding_config,
                 recurrence_decoding_config,
                 command_decoding_config,
                 device,
                 trained_weight,
                 cfg_command):

        super(CVAE_implicit_distribution_inference, self).__init__()

        self.state_encoding_config = state_encoding_config
        self.latent_decoding_config = latent_decoding_config
        self.recurrence_decoding_config = recurrence_decoding_config
        self.command_decoding_config = command_decoding_config
        self.device = device
        self.trained_weight = trained_weight
        self.cfg_command = cfg_command
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
        self.latent_dim = self.latent_decoding_config["input"] - self.state_encoding_config["output"]

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.recurrence_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        self.set_module()

    def set_module(self):
        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.latent_decoder = MLP(self.latent_decoding_config["shape"],
                                  self.activation_map[self.latent_decoding_config["activation"]],
                                  self.latent_decoding_config["input"],
                                  self.latent_decoding_config["output"],
                                  dropout=self.state_encoding_config["dropout"],
                                  batchnorm=self.state_encoding_config["batchnorm"])
        self.recurrence_decoder = torch.nn.GRU(self.recurrence_decoding_config["input"],
                                               self.recurrence_decoding_config["hidden"],
                                               self.recurrence_decoding_config["layer"])
        self.command_decoder = MLP(self.command_decoding_config["shape"],
                                   self.activation_map[self.command_decoding_config["activation"]],
                                   self.command_decoding_config["input"],
                                   self.command_decoding_config["output"],
                                   dropout=self.state_encoding_config["dropout"],
                                   batchnorm=self.state_encoding_config["batchnorm"])

        # load weight
        state_encoder_state_dict = self.state_encoder.state_dict()
        trained_state_encoder_state_dict = {k: v for k, v in self.trained_weight.items() if k in state_encoder_state_dict}
        state_encoder_state_dict.update(trained_state_encoder_state_dict)
        self.state_encoder.load_state_dict(state_encoder_state_dict)
        self.state_encoder.eval()

        latent_decoder_state_dict = self.latent_decoder.state_dict()
        trained_latent_decoder_state_dict = {k: v for k, v in self.trained_weight.items() if k in latent_decoder_state_dict}
        latent_decoder_state_dict.update(trained_latent_decoder_state_dict)
        self.latent_decoder.load_state_dict(latent_decoder_state_dict)
        self.latent_decoder.eval()

        recurrence_decoder_state_dict = self.recurrence_decoder.state_dict()
        trained_recurrence_decoder_state_dict = {k: v for k, v in self.trained_weight.items() if k in recurrence_decoder_state_dict}
        recurrence_decoder_state_dict.update(trained_recurrence_decoder_state_dict)
        self.recurrence_decoder.load_state_dict(recurrence_decoder_state_dict)
        self.recurrence_decoder.eval()

        command_decoder_state_dict = self.command_decoder.state_dict()
        trained_command_decoder_state_dict = {k: v for k, v in self.trained_weight.items() if k in command_decoder_state_dict}
        command_decoder_state_dict.update(trained_command_decoder_state_dict)
        self.command_decoder.load_state_dict(command_decoder_state_dict)
        self.command_decoder.eval()

    def forward(self, state, n_sample, traj_len):
        """

        :param state: (n_sample, state_dim)
        :param n_sample: int
        :param traj_len: int
        :return:
        """
        with torch.no_grad():
            encoded_state = self.state_encoder.architecture(state)
            sample = torch.rand((n_sample, self.latent_dim),
                                 dtype=encoded_state.type, layout=encoded_state.layout, device=self.device)

            # decode command trajectory
            total_decoded_result = torch.cat((encoded_state, sample), dim=1)
            hidden_state = self.latent_decoder.architecture(total_decoded_result).unsqueeze(0)
            decoded_traj = torch.zeros(traj_len, n_sample, self.recurrence_decoding_config["hidden"]).to(self.device)
            input = torch.zeros(1, n_sample, self.recurrence_decoding_config["input"]).to(self.device)

            for i in range(traj_len):
                output, hidden_state = self.recurrence_decoder(input, hidden_state)
                output = output.squeeze(0)
                decoded_traj[i] = output
                input = decoded_traj[i]

            decoded_traj = decoded_traj.reshape(-1, self.recurrence_decoding_config["hidden"])
            sampled_command_traj = self.command_decoder.architecture(decoded_traj)
            sampled_command_traj = sampled_command_traj.reshape(traj_len, n_sample, -1)

            sampled_command_traj = sampled_command_traj.cpu().detach().numpy()
            sampled_command_traj = np.clip(sampled_command_traj,
                                           [self.cfg_command["forward_vel"]["min"], self.cfg_command["lateral_vel"]["min"], self.cfg_command["yaw_rate"]["min"]],
                                           [self.cfg_command["forward_vel"]["max"], self.cfg_command["lateral_vel"]["max"], self.cfg_command["yaw_rate"]["max"]])

            return sampled_command_traj


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
