import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb

class Chompld(nn.Module):
    def __init__(self, chomp_size):
        super(Chompld, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, activation='relu'):
        super(TemporalBlock, self).__init__()
        self.activation_map = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "leakyrelu": nn.LeakyReLU()}
        assert activation in list(self.activation_map.keys()), "Unavailable activation."

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chompld(padding)
        self.activation1 = self.activation_map[activation]
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chompld(padding)
        self.activation2 = self.activation_map[activation]
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.activation1, self.dropout1,
                                 self.conv2, self.chomp2, self.activation2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.activation = self.activation_map[activation]
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dropout=0.2, activation='relu'):
        """

        Implementation of "Temporal Convolutional Network (TCN)"

        https://sanghyu.tistory.com/24
        https://hongl.tistory.com/253
        https://github.com/philipperemy/keras-tcn

        ==============================================================

        Receptive_field_size = 1 + (kernel_size - 1) * sum(dilation)

        ex)
        kernel_size = 3, dilation = [1, 1] ==> Receptive_field_size = 5
        kernel_size = 3, dilation = [1, 1, 3, 3] ==> Receptive_field_size = 17
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout, activation=activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x, only_last=True):
        """

        :param x: (n_batch, feature_dim, time_step)
        :param only_last: whether to return only last step encoded result or not
        :return:
            - only_last = True ==> (n_batch, feature_dim)
            - only_last = False ==> (n_batch, feature_dim, time_step)
        """
        if only_last:
            return self.network(x)[:, :, -1]
        else:
            return self.network(x)

# tcn = TemporalConvNet(6, [16, 16, 16], kernel_size=5, stride=1, dropout=0.2)
# input = torch.rand(10, 6, 57)
# output = tcn(input, only_last=True)
# print(output.shape)
# pytorch_total_params = sum(p.numel() for p in tcn.parameters())
# print(pytorch_total_params)

