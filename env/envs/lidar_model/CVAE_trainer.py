import torch
import numpy as np
import os
from torch.utils.data import Dataset

class CVAE_dataset(Dataset):
    def __init__(self, data_file_list, file_idx_list, folder_path):
        """

        :param data_file_list: list of file names
        :param file_idx_list: list of file idx (based on the data_file_list)
        :param folder_path: FULL path to the data directory without '/' at last
        """
        self.data_file_list = data_file_list
        self.file_idx_list = file_idx_list
        self.data_folder_path = folder_path

    def __len__(self):
        return len(self.file_idx_list)

    def __getitem__(self, index):
        sampled_data_file = self.data_file_list[self.file_idx_list[index]]
        sampled_data_file_path = f"{self.data_folder_path}/{sampled_data_file}"
        sampled_data = np.load(sampled_data_file_path)
        observation = torch.from_numpy(sampled_data["observation"].astype(np.float32))
        goal_position = torch.from_numpy(sampled_data["goal_position"].astype(np.float32))
        command_traj = torch.from_numpy(sampled_data["command_traj"].astype(np.float32))

        return observation, goal_position, command_traj

