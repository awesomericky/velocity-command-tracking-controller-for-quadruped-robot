import time
from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from raisimGymTorch.env.envs.lidar_model.model import CVAE_implicit_distribution, CVAE_implicit_distribution_inference
from raisimGymTorch.env.envs.lidar_model.CVAE_trainer import CVAE_dataset
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import wandb
import pdb
import os
import argparse
import numpy as np

# task specification
task_name = "CVAE_train"

parser = argparse.ArgumentParser()
parser.add_argument('-pw', '--pretrained_weight', help='pre-trained weight path for state and command encoder', type=str,
                    required=True)
# pretrained_weight example format
# /home/student/quadruped/raisimLib/raisimGymTorch/data/lidar_environment_model/2021-11-09-08-00-33/full_2300.pt
args = parser.parse_args()

pretrained_weight = args.pretrained_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark = True

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# set numpy seed (crucial for creating train & validationm data)
seed = cfg["CVAE_training"]["seed"]
np.random.seed(seed)

# Create dataset
data_folder_path = "/home/student/quadruped/raisimLib/raisimGymTorch/CVAE_data"
data_file_list = os.listdir(data_folder_path)
n_data_files = len(data_file_list)
train_data_ratio = 0.9
n_train_data_files = int(n_data_files * train_data_ratio)
n_val_data_files = n_data_files - n_train_data_files
indices = np.random.permutation(n_data_files)
train_idx, validation_idx = indices[:n_train_data_files], indices[n_train_data_files:]
print("<--- Dataset size --->")
print(f"Train: {len(train_idx)} / Validation: {len(validation_idx)}")
print("----------------------")

training_set = CVAE_dataset(data_file_list=data_file_list, file_idx_list=train_idx, folder_path=data_folder_path)
training_generator = DataLoader(training_set,
                                batch_size=cfg["CVAE_training"]["batch_size"],
                                shuffle=cfg["CVAE_training"]["shuffle_batch"],
                                num_workers=cfg["CVAE_training"]["num_workers"],
                                drop_last=True)

validation_set = CVAE_dataset(data_file_list=data_file_list, file_idx_list=validation_idx, folder_path=data_folder_path)
validation_generator = DataLoader(validation_set,
                                  batch_size=cfg["CVAE_training"]["batch_size"],
                                  shuffle=cfg["CVAE_training"]["shuffle_batch"],
                                  num_workers=cfg["CVAE_training"]["num_workers"],
                                  drop_last=True)

# Create CVAE training model
cvae_train_model = CVAE_implicit_distribution(state_encoding_config=cfg["CVAE_architecture"]["state_encoder"],
                                              command_encoding_config=cfg["CVAE_architecture"]["command_encoder"],
                                              recurrence_encoding_config=cfg["CVAE_architecture"]["recurrence_encoder"],
                                              latent_encoding_config=cfg["CVAE_architecture"]["latent_encoder"],
                                              latent_decoding_config=cfg["CVAE_architecture"]["latent_decoder"],
                                              recurrence_decoding_config=cfg["CVAE_architecture"]["recurrence_decoder"],
                                              command_decoding_config=cfg["CVAE_architecture"]["command_decoder"],
                                              device=device,
                                              pretrained_weight=pretrained_weight,
                                              n_latent_sample=cfg["CVAE_training"]["n_latent_sample"],
                                              state_encoder_fixed=True,
                                              command_encoder_fixed=True)
cvae_train_model.to(device)
n_latent_sample = cfg["CVAE_training"]["n_latent_sample"]
n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
loss_weight = {"reconstruction": cfg["CVAE_training"]["loss_weight"]["reconsturction"],
               "KL_posterior": cfg["CVAE_training"]["loss_weight"]["KL_posterior"]}

optimizer = optim.Adam(filter(lambda p: p.requires_grad, cvae_train_model.parameters()), lr=cfg["CVAE_training"]["learning_rate"])

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

if cfg["logging"]:
    wandb.init(name=task_name, project="Quadruped_RL")
    wandb.watch(cvae_train_model, log='all', log_freq=150)

pdb.set_trace()

for epoch in range(cfg["CVAE_training"]["num_epochs"]):
    if epoch % cfg["CVAE_training"]["evaluate_period"] == 0:
        print("Evaluating the current CVAE model")
        torch.save({
            'model_architecture_state_dict': cvae_train_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(epoch)+'.pt')

        # Create CVAE evaluating model (create new graph)
        cvae_evaluate_model = CVAE_implicit_distribution(state_encoding_config=cfg["CVAE_architecture"]["state_encoder"],
                                                         command_encoding_config=cfg["CVAE_architecture"]["command_encoder"],
                                                         recurrence_encoding_config=cfg["CVAE_architecture"]["recurrence_encoder"],
                                                         latent_encoding_config=cfg["CVAE_architecture"]["latent_encoder"],
                                                         latent_decoding_config=cfg["CVAE_architecture"]["latent_decoder"],
                                                         recurrence_decoding_config=cfg["CVAE_architecture"]["recurrence_decoder"],
                                                         command_decoding_config=cfg["CVAE_architecture"]["command_decoder"],
                                                         device=device,
                                                         pretrained_weight=pretrained_weight,
                                                         n_latent_sample=cfg["CVAE_training"]["n_latent_sample"],
                                                         state_encoder_fixed=True,
                                                         command_encoder_fixed=True)
        cvae_evaluate_model.load_state_dict(torch.load(saver.data_dir+"/full_"+str(epoch)+'.pt', map_location=device)['model_architecture_state_dict'])
        cvae_evaluate_model.eval()
        cvae_evaluate_model.to(device)

        # Create CVAE inference model
        saved_model_weight = saver.data_dir+"/full_"+str(epoch)+'.pt'
        cvae_inference_model = CVAE_implicit_distribution_inference(state_encoding_config=cfg["CVAE_architecture"]["state_encoder"],
                                                                    latent_decoding_config=cfg["CVAE_architecture"]["latent_decoder"],
                                                                    recurrence_decoding_config=cfg["CVAE_architecture"]["recurrence_decoder"],
                                                                    command_decoding_config=cfg["CVAE_architecture"]["command_decoder"],
                                                                    device=device,
                                                                    trained_weight=saved_model_weight,
                                                                    cfg_command=cfg["environment"]["command"])
        cvae_inference_model.eval()
        cvae_inference_model.to(device)

        mean_loss = 0
        mean_reconstruction_loss = 0
        mean_KL_posterior_loss = 0
        mean_inference_reconstruction_loss = 0
        n_update = 0

        for observation_batch, goal_position_batch, command_traj_batch in validation_generator:
            observation_batch = observation_batch.to(device)
            goal_position_batch = goal_position_batch.to(device)
            command_traj_batch = torch.swapaxes(command_traj_batch, 0, 1)
            command_traj_batch = command_traj_batch.to(device)

            # Model forward computation
            with torch.no_grad():
                latent_mean, latent_log_var, sampled_command_traj = cvae_evaluate_model(observation_batch, goal_position_batch, command_traj_batch)
                inference_sampled_command_traj = cvae_inference_model(observation_batch, goal_position_batch, cfg["CVAE_inference"]["n_sample"], n_prediction_step, return_torch=True) 
            
            # Compute loss
            if n_latent_sample == 1:
                reconstruction_loss = torch.sum(torch.sum((sampled_command_traj - command_traj_batch).pow(2), dim=0), dim=-1)
            else:
                command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                                  (command_traj_batch.shape[0], command_traj_batch.shape[1], n_latent_sample, command_traj_batch.shape[2]))
                if cfg["CVAE_training"]["objective_type"] == "CVAE":
                    reconstruction_loss = torch.mean(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)
                elif cfg["CVAE_training"]["objective_type"] == "BMS":
                    reconstruction_loss = torch.min(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)[0] + torch.log(torch.tensor(n_latent_sample)).to(device)
                else:
                    raise ValueError("Unsupported loss type")
            reconstruction_loss = reconstruction_loss.mean()
            KL_posterior_loss = 0.5 * (torch.sum(latent_mean.pow(2) + latent_log_var.exp() - latent_log_var - 1, dim=-1))
            KL_posterior_loss = KL_posterior_loss.mean()
            loss = reconstruction_loss * loss_weight["reconstruction"] + KL_posterior_loss * loss_weight["KL_posterior"]

            
            command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                            (command_traj_batch.shape[0], command_traj_batch.shape[1], cfg["CVAE_inference"]["n_sample"], command_traj_batch.shape[2]))
            inference_reconstruction_loss = torch.mean(torch.sum(torch.sum((inference_sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1).mean()

            mean_loss += loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_KL_posterior_loss += KL_posterior_loss.item()
            mean_inference_reconstruction_loss += inference_reconstruction_loss.item()
            n_update += 1

        mean_loss /= n_update
        mean_reconstruction_loss /= n_update
        mean_KL_posterior_loss /= n_update
        mean_inference_reconstruction_loss /= n_update

        if cfg["logging"]:
            # Log data
            logging_data = dict()
            logging_data['Evaluate/Total'] = mean_loss
            logging_data['Evaluate/Reconstruction'] = mean_reconstruction_loss
            logging_data['Evaluate/KL_posterior'] = mean_KL_posterior_loss
            logging_data['Evaluate/Inference_reconstruction'] = mean_inference_reconstruction_loss
            wandb.log(logging_data)

        print('====================================================')
        print('{:>6}th evaluation'.format(epoch))
        print('{:<40} {:>6}'.format("total: ", '{:0.6f}'.format(mean_loss)))
        print('{:<40} {:>6}'.format("reconstruction: ", '{:0.6f}'.format(mean_reconstruction_loss)))
        print('{:<40} {:>6}'.format("kl posterior: ", '{:0.6f}'.format(mean_KL_posterior_loss)))
        print('{:<40} {:>6}'.format("inference reconstruction: ", '{:0.6f}'.format(mean_inference_reconstruction_loss)))

        print('====================================================\n')
    

    epoch_start = time.time()

    mean_loss = 0
    mean_reconstruction_loss = 0
    mean_KL_posterior_loss = 0
    n_update = 0

    for observation_batch, goal_position_batch, command_traj_batch in training_generator:
        observation_batch = observation_batch.to(device)
        goal_position_batch = goal_position_batch.to(device)
        command_traj_batch = torch.swapaxes(command_traj_batch, 0, 1)
        command_traj_batch = command_traj_batch.to(device)

        # Model forward computation
        latent_mean, latent_log_var, sampled_command_traj = cvae_train_model(observation_batch, goal_position_batch, command_traj_batch)

        # Compute loss
        if n_latent_sample == 1:
            reconstruction_loss = torch.sum(torch.sum((sampled_command_traj - command_traj_batch).pow(2), dim=0), dim=-1)
        else:
            command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                              (command_traj_batch.shape[0], command_traj_batch.shape[1], n_latent_sample, command_traj_batch.shape[2]))
            if cfg["CVAE_training"]["objective_type"] == "CVAE":
                reconstruction_loss = torch.mean(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)
            elif cfg["CVAE_training"]["objective_type"] == "BMS":
                reconstruction_loss = torch.min(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)[0] +  torch.log(torch.tensor(n_latent_sample)).to(device)
            else:
                raise ValueError("Unsupported loss type")
        reconstruction_loss = reconstruction_loss.mean()
        KL_posterior_loss = 0.5 * (torch.sum(latent_mean.pow(2) + latent_log_var.exp() - latent_log_var - 1, dim=-1))
        KL_posterior_loss = KL_posterior_loss.mean()
        loss = reconstruction_loss * loss_weight["reconstruction"] + KL_posterior_loss * loss_weight["KL_posterior"]

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        if cfg["CVAE_training"]["clip_gradient"]:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, cvae_train_model.parameters()), cfg["CVAE_training"]["max_gradient_norm"])
        optimizer.step()

        mean_loss += loss.item()
        mean_reconstruction_loss += reconstruction_loss.item()
        mean_KL_posterior_loss += KL_posterior_loss.item()
        n_update += 1

    mean_loss /= n_update
    mean_reconstruction_loss /= n_update
    mean_KL_posterior_loss /= n_update

    if cfg["logging"]:
        # Log data
        logging_data = dict()
        logging_data['Loss/Total'] = mean_loss
        logging_data['Loss/Reconstruction'] = mean_reconstruction_loss
        logging_data['Loss/KL_posterior'] = mean_KL_posterior_loss
        wandb.log(logging_data)

    epoch_end = time.time()
    elapse_time_seconds = epoch_end - epoch_start
    elaspe_time_minutes = int(elapse_time_seconds / 60)
    elapse_time_seconds -= (elaspe_time_minutes * 60)
    elapse_time_seconds = int(elapse_time_seconds)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(epoch))
    print('{:<40} {:>6}'.format("total: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("reconstruction: ", '{:0.6f}'.format(mean_reconstruction_loss)))
    print('{:<40} {:>6}'.format("kl posterior: ", '{:0.6f}'.format(mean_KL_posterior_loss)))
    print(f'Time: {elaspe_time_minutes}m {elapse_time_seconds}s')
    print('----------------------------------------------------\n')




