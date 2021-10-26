from shutil import copyfile
import datetime
import os
import ntpath
import torch
import numpy as np


class ConfigurationSaver:
    def __init__(self, log_dir, save_items):
        self._data_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir
        

def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)


def load_param(weight_path, env, actor, critic, optimizer, data_dir):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    mean_csv_path = weight_dir + 'mean' + iteration_number + '.csv'
    var_csv_path = weight_dir + 'var' + iteration_number + '.csv'
    items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def load_enviroment_model_param(weight_path, model, optimizer, data_dir):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nRetraining from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    items_to_save = [weight_path, weight_dir + "cfg.yaml", weight_dir + "Environment.hpp"]

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load environment model parameters from full checkpoint
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class UserCommand:
    def __init__(self, cfg, n_envs):
        self.min_forward_vel = cfg['environment']['command']['forward_vel']['min']
        self.max_forward_vel = cfg['environment']['command']['forward_vel']['max']
        self.min_lateral_vel = cfg['environment']['command']['lateral_vel']['min']
        self.max_lateral_vel = cfg['environment']['command']['lateral_vel']['max']
        self.min_yaw_rate = cfg['environment']['command']['yaw_rate']['min']
        self.max_yaw_rate = cfg['environment']['command']['yaw_rate']['max']
        self.n_envs = n_envs
    
    def uniform_sample_train(self):
        forward_vel = np.random.uniform(low=self.min_forward_vel, high=self.max_forward_vel, size=self.n_envs)
        lateral_vel = np.random.uniform(low=self.min_lateral_vel, high=self.max_lateral_vel, size=self.n_envs)
        yaw_rate = np.random.uniform(low=self.min_yaw_rate, high=self.max_yaw_rate, size=self.n_envs)
        command = np.stack((forward_vel, lateral_vel, yaw_rate), axis=1)
        return np.ascontiguousarray(command).astype(np.float32)
    
    def uniform_sample_evaluate(self):
        forward_vel = np.random.uniform(low=self.min_forward_vel, high=self.max_forward_vel, size=1)
        lateral_vel = np.random.uniform(low=self.min_lateral_vel, high=self.max_lateral_vel, size=1)
        yaw_rate = np.random.uniform(low=self.min_yaw_rate, high=self.max_yaw_rate, size=1)
        command = np.stack((forward_vel, lateral_vel, yaw_rate), axis=1)
        return np.ascontiguousarray(np.broadcast_to(command, (self.n_envs, 3))).astype(np.float32)