import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
import pickle
import tqdm
import os
from softgym.utils.visualization import save_numpy_as_gif
from sb3.utils import make_dir
from perceiver_pytorch import PerceiverIO

'''
1. Pre-trains a "forward" dynamics model on random action dataset.
2. Freeze "forward" dynamics model and train an inverse dynamics model on random action dataset.
'''

class InverseDynamicsModelLSTM(nn.Module):
    """LSTM implementation"""

    def __init__(self, obs_dim, act_dim, hidden_size=32, num_layers=1):
        super(InverseDynamicsModelLSTM, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(obs_dim * 2, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.LayerNorm(act_dim))


    def forward(self, x, hs, target_shape):
        out, hs = self.lstm(x, hs)              # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)     
        out = self.head(out)                    # out.shape = (batch_size * seq_len, act_dim)
        out = out.reshape(target_shape)
        out = torch.tanh(out)
        return out, hs

class MultiStepForwardDynamicsLSTM(nn.Module):
    """LSTM implementation"""

    def __init__(self, obs_dim, act_dim, hidden_size=32, num_layers=1):
        super(MultiStepForwardDynamicsLSTM, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(obs_dim + act_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, obs_dim),
            nn.LayerNorm(obs_dim))

    def forward(self, x, hs, target_shape):
        out, hs = self.lstm(x, hs)               # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)
        out = self.head(out)                    # out.shape = (batch_size * seq_len, obs_dim)
        out = out.reshape(target_shape)
        out = torch.tanh(out)
        return out, hs

class ForwardDynamicsLSTMReducedObs(nn.Module):
    """Particle-based LSTM model that outputs predicted 4 corner positions of a cloth"""

    def __init__(self, obs_dim, act_dim, num_particles, hidden_size=32, num_layers=1):
        super(ForwardDynamicsLSTMReducedObs, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.num_features = 3
        self.cloth_dim = num_particles
        self.lstm = nn.LSTM((self.cloth_dim * self.num_features) + act_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, obs_dim),
            nn.LayerNorm(obs_dim))

    def forward(self, x, acts, hs, target_shape):
        x = torch.flatten(x, 2)
        out = torch.cat((x, acts), axis=2)
        out, hs = self.lstm(out, hs)               # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)
        out = self.head(out)
        out = out.reshape(target_shape)
        out = torch.tanh(out)
        return out, hs

class ForwardDynamicsLSTMParticles(nn.Module):
    """Particle-based LSTM model that outputs predicted particle positions"""

    def __init__(self, obs_dim, act_dim, num_particles, hidden_size=32, num_layers=1):
        super(ForwardDynamicsLSTMParticles, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.num_features = 3
        self.cloth_dim = num_particles
        self.total_cloth_features = self.cloth_dim * self.num_features
        self.lstm = nn.LSTM(self.total_cloth_features + act_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, self.total_cloth_features),
            nn.LayerNorm(self.total_cloth_features))

    def forward(self, x, acts, hs, target_shape):
        x = torch.flatten(x, 2)
        out = torch.cat((x, acts), axis=2)
        out, hs = self.lstm(out, hs)               # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)
        out = self.head(out)
        out = out.reshape(target_shape)
        out = torch.tanh(out)
        return out, hs

class CNNForwardDynamics(nn.Module):
    """Particle-based CNN model"""

    def __init__(self, obs_dim, act_dim, num_particles, hidden_size=32, num_layers=1):
        super(CNNForwardDynamics, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.num_features = 3
        self.cloth_dim = num_particles
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.num_features, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )
        test_mat = torch.zeros(1, self.num_features, int(np.sqrt(self.cloth_dim)), int(np.sqrt(self.cloth_dim)))
        for conv_layer in self.cnn_layers:
            test_mat = conv_layer(test_mat)
        fc_input_size = int(np.prod(test_mat.shape))

        self.head = nn.Sequential(
            nn.Linear(fc_input_size + act_dim, self.cloth_dim * self.num_features),
            nn.LayerNorm(self.cloth_dim * self.num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, acts, target_shape):
        out = self.cnn_layers(x)
        out = torch.flatten(out, 1)
        out = torch.cat((out, acts), axis=1)
        out = self.head(out)
        out = torch.tanh(out)
        out = out.reshape(target_shape)
        return out

class CNNForwardDynamicsLSTM(nn.Module):
    """Particle-based CNN LSTM model"""

    def __init__(self, obs_dim, act_dim, num_particles, particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=32, num_layers=1, env_kwargs=None):
        super(CNNForwardDynamicsLSTM, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.is_three_cubes_env = env_kwargs is not None and env_kwargs['env'] == 'ThreeCubes'
        if self.is_three_cubes_env:
            self.num_features = 1
        else:
            self.num_features = 3
        self.cloth_dim = num_particles
        self.particle_based_cnn_lstm_fwd_dyn_mode = particle_based_cnn_lstm_fwd_dyn_mode

        if self.particle_based_cnn_lstm_fwd_dyn_mode == '1dconv':
            # 1D Conv
            if self.is_three_cubes_env:
                self.cnn_layers = nn.Sequential(
                    nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                )
            else:
                self.cnn_layers = nn.Sequential(
                    nn.Conv1d(in_channels=self.num_features, out_channels=32, kernel_size=3, stride=2),
                    nn.LeakyReLU(),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                    nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                )
            test_mat = torch.zeros(1, self.num_features, self.cloth_dim)
        elif self.particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
            # 2D Conv
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(in_channels=self.num_features, out_channels=32, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
                nn.LeakyReLU(),
            )
            test_mat = torch.zeros(1, self.num_features, int(np.sqrt(self.cloth_dim)), int(np.sqrt(self.cloth_dim)))

        for conv_layer in self.cnn_layers:
            test_mat = conv_layer(test_mat)
        fc_input_size = int(np.prod(test_mat.shape))

        self.lstm = nn.LSTM(fc_input_size + act_dim, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, self.cloth_dim * self.num_features),
            nn.LayerNorm(self.cloth_dim * self.num_features))

    def forward(self, x, acts, hs, target_shape):
        if self.is_three_cubes_env:
            out = self.cnn_layers(x)
            out = out[:, None, :, :]
            out = torch.reshape(out, (out.shape[0], out.shape[1], int(out.shape[2] * out.shape[3]))) # 1D Conv
        else:
            for time_sequence_t in range(x.shape[1]):
                if time_sequence_t == 0:
                    out = self.cnn_layers(x[:, time_sequence_t, :, :])
                    out = out[:, None, :, :]
                else:
                    cnn_out = self.cnn_layers(x[:, time_sequence_t, :, :])
                    cnn_out = cnn_out[:, None, :, :]
                    out = torch.cat((out, cnn_out), axis=1)

            # squeeze the last dimension so it's (batch_size, time_sequence, features)
            if self.particle_based_cnn_lstm_fwd_dyn_mode == '1dconv':
                out = torch.reshape(out, (out.shape[0], out.shape[1], int(out.shape[2] * out.shape[3]))) # 1D Conv
            elif self.particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
                out = torch.reshape(out, (out.shape[0], out.shape[1], int(out.shape[2] * out.shape[3] * out.shape[4]))) # 2D Conv
        out = torch.cat((out, acts), axis=2)
        out, hs = self.lstm(out, hs)               # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)
        out = self.head(out)
        out = out.reshape(target_shape)
        out = torch.tanh(out)
        return out, hs

class PerceiverIOForwardDynamics(nn.Module):
    """
    Perceiver IO
    Implementation from # https://github.com/lucidrains/perceiver-pytorch
    """

    def __init__(self, act_dim, num_particles, env_kwargs=None):
        super(PerceiverIOForwardDynamics, self).__init__()
        # Build the model
        self.is_three_cubes_env = env_kwargs is not None and env_kwargs['env'] == 'ThreeCubes'
        if self.is_three_cubes_env:
            raise NotImplementedError()
        else:
            self.num_features = 3
        self.cloth_dim = num_particles
        self.cloth_features_dim = num_particles * self.num_features
        self.cloth_act_dim = num_particles + int(act_dim / self.num_features)
        self.logits_dim = 512
        self.logits_features_dim = int(self.logits_dim * self.num_features)

        self.perceiverio = PerceiverIO(
            dim = self.cloth_act_dim,                  # dimension of sequence to be encoded
            queries_dim = self.cloth_act_dim,          # dimension of decoder queries
            logits_dim = self.cloth_dim,               # dimension of final logits
            depth = 6,                                 # depth of net
            num_latents = 256,                         # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 256,                          # latent dimension
            cross_heads = 1,                           # number of heads for cross attention. paper said 1
            latent_heads = 8,                          # number of heads for latent self attention, 8
            cross_dim_head = 64,                       # number of dimensions per cross attention head
            latent_dim_head = 64,                      # number of dimensions per latent self attention head
            weight_tie_layers = False,                 # whether to weight tie layers (optional, as indicated in the diagram)
            seq_dropout_prob = 0.2                     # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
        )
        # self.head = nn.Sequential(
        #     nn.Linear(self.logits_features_dim, self.cloth_features_dim),
        #     nn.LayerNorm(self.cloth_features_dim))

    def forward(self, x, acts, target_shape):
        if len(x.shape) == 4 and len(acts.shape) == 3:
            x = x[:, 0, :, :]
            acts = acts[:, 0, :]
        acts = acts.reshape(acts.shape[0], self.num_features, -1)
        x = torch.cat((x, acts), axis=2)
        logits = self.perceiverio(x, queries=x)
        # logits = logits.reshape(logits.shape[0], -1)
        # logits = self.head(logits)
        logits = logits.reshape(target_shape)
        out = torch.tanh(logits)
        return out

class ExpertDemonstrations(Dataset):
    def __init__(self, args, num_action_vals_two_pickers):
        self.num_action_vals_two_pickers = num_action_vals_two_pickers
        self.enable_particle_based_fwd_dyn = args.get('enable_particle_based_fwd_dyn')
        self.data = self.load_file(args['two_arms_expert_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, goal_ob, action = self.data["obs"][idx], self.data["goal_obs"][idx], self.data["acts"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'goal_ob': torch.from_numpy(goal_ob),
            'action': torch.from_numpy(action),
        }

        if self.enable_particle_based_fwd_dyn:
            particles  = self.data["particles"][idx]
            out['particles'] = torch.from_numpy(particles)
            goal_delta_pos_particles  = self.data["goal_delta_pos_particles"][idx]
            out['goal_delta_pos_particles'] = torch.from_numpy(goal_delta_pos_particles)

        return out

    def load_file(self, file_path):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'goal_obs': [],
            'acts': [],
            'particles': [],
            'goal_delta_pos_particles': [],
        }
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            ob_trajs = data['ob_trajs']
            next_obs_trajs = data['ob_next_trajs']
            action_trajs = data['action_trajs']

            final_data['obs'] = ob_trajs[:, 0, :-self.num_action_vals_two_pickers]
            final_data['goal_obs'] = next_obs_trajs[:, 0, :-self.num_action_vals_two_pickers]
            final_data['acts'] = action_trajs[:, 0, :-self.num_action_vals_two_pickers]
            final_data['obs'] = final_data['obs'][:, None, :]
            final_data['goal_obs'] = final_data['goal_obs'][:, None, :]
            final_data['acts'] = final_data['acts'][:, None, :]

            if self.enable_particle_based_fwd_dyn:
                particles_trajs = data['particles_trajs'][:, 0, :, :][:, None, :, :]
                particles_trajs = np.transpose(particles_trajs, (0, 1, 3, 2))
                final_data['particles'] = particles_trajs

                particles_next_trajs = data['particles_next_trajs'][:, 0, :, :][:, None, :, :]
                particles_next_trajs = np.transpose(particles_next_trajs, (0, 1, 3, 2))
                final_data['goal_delta_pos_particles'] = particles_next_trajs - particles_trajs

        print('finished loading data.')
        return final_data

class DemonstrationsNonLSTM(Dataset):
    def __init__(self, args, num_action_vals_one_picker, downsample_idx):
        self.num_action_vals_one_picker = num_action_vals_one_picker
        self.enable_particle_based_fwd_dyn = args.get('enable_particle_based_fwd_dyn')
        self.extract_num_time_step_data = args.get('extract_num_time_step_data')
        self.downsample_idx = downsample_idx
        self.data = self.load_file(args['random_actions_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, goal_ob, action  = self.data["obs"][idx], self.data["goal_obs"][idx], self.data["acts"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'goal_ob': torch.from_numpy(goal_ob),
            'action': torch.from_numpy(action),
        }

        if self.enable_particle_based_fwd_dyn:
            particles  = self.data["particles"][idx]
            out['particles'] = torch.from_numpy(particles)
            goal_delta_pos_particles  = self.data["goal_delta_pos_particles"][idx]
            out['goal_delta_pos_particles'] = torch.from_numpy(goal_delta_pos_particles)

        return out

    def load_file(self, file_paths):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'goal_obs': [],
            'acts': [],
            'particles': [],
            'goal_delta_pos_particles': [],
        }

        if len(file_paths) > 1:
            raise NotImplementedError

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                ob_trajs = data['ob_trajs']
                next_obs_trajs = data['ob_next_trajs']
                action_trajs = data['action_trajs']

                final_data['obs'] = ob_trajs[:, 0, :-self.num_action_vals_one_picker]
                final_data['goal_obs'] = next_obs_trajs[:, -1, :-self.num_action_vals_one_picker]
                final_data['acts'] = action_trajs[:, 0, :]

                if self.enable_particle_based_fwd_dyn:

                    particles_trajs = data['particles_trajs'][:, 0, :, :]
                    particles_next_trajs = data['particles_next_trajs'][:, 0, :, :]

                    if self.downsample_idx is not None:
                        particles_trajs = particles_trajs[:, self.downsample_idx, :]
                        particles_next_trajs = particles_next_trajs[:, self.downsample_idx, :]

                    particles_trajs = np.transpose(particles_trajs, (0, 2, 1))
                    # reshape into [batch, xyz, width_cloth, height_cloth]
                    particles_trajs = np.reshape(particles_trajs, (particles_trajs.shape[0], particles_trajs.shape[1], int(np.sqrt(particles_trajs.shape[2])), int(np.sqrt(particles_trajs.shape[2]))))
                    final_data['particles'] = particles_trajs

                    particles_next_trajs = np.transpose(particles_next_trajs, (0, 2, 1))
                    # reshape into [batch, xyz, width_cloth, height_cloth]
                    particles_next_trajs = np.reshape(particles_next_trajs, (particles_next_trajs.shape[0], particles_next_trajs.shape[1], int(np.sqrt(particles_next_trajs.shape[2])), int(np.sqrt(particles_next_trajs.shape[2]))))
                    final_data['goal_delta_pos_particles'] = particles_next_trajs - particles_trajs

        print('finished loading data.')
        return final_data

class Demonstrations(Dataset):
    def __init__(self, args, num_action_vals_one_picker, downsample_idx, particle_based_cnn_lstm_fwd_dyn_mode, env_kwargs=None):
        self.num_action_vals_one_picker = num_action_vals_one_picker
        self.enable_particle_based_fwd_dyn = args.get('enable_particle_based_fwd_dyn')
        self.extract_num_time_step_data = args.get('extract_num_time_step_data')
        self.downsample_idx = downsample_idx
        self.particle_based_cnn_lstm_fwd_dyn_mode = particle_based_cnn_lstm_fwd_dyn_mode
        self.is_three_cubes_env = env_kwargs is not None and env_kwargs['env'] == 'ThreeCubes'
        self.data = self.load_file(args['random_actions_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, goal_ob, action  = self.data["obs"][idx], self.data["goal_obs"][idx], self.data["acts"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'goal_ob': torch.from_numpy(goal_ob),
            'action': torch.from_numpy(action),
        }

        if self.enable_particle_based_fwd_dyn:
            particles  = self.data["particles"][idx]
            out['particles'] = torch.from_numpy(particles)
            goal_delta_pos_particles  = self.data["goal_delta_pos_particles"][idx]
            out['goal_delta_pos_particles'] = torch.from_numpy(goal_delta_pos_particles)

        return out

    def load_file(self, file_paths):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'goal_obs': [],
            'acts': [],
            'particles': [],
            'goal_delta_pos_particles': [],
        }

        if len(file_paths) > 1:
            raise NotImplementedError

        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                ob_trajs = data['ob_trajs'][:, :self.extract_num_time_step_data, :] if self.extract_num_time_step_data else data['ob_trajs']
                next_obs_trajs = data['ob_next_trajs'][:, :self.extract_num_time_step_data, :] if self.extract_num_time_step_data else data['ob_next_trajs']
                action_trajs = data['action_trajs'][:, :self.extract_num_time_step_data, :] if self.extract_num_time_step_data else data['action_trajs']

                if self.is_three_cubes_env:
                    final_data['obs'] = ob_trajs
                    final_data['goal_obs'] = next_obs_trajs
                    final_data['acts'] = action_trajs
                else:
                    final_data['obs'] = ob_trajs[:, :, :-self.num_action_vals_one_picker]
                    final_data['goal_obs'] = next_obs_trajs[:, -1, :-self.num_action_vals_one_picker][:, None, :]
                    final_data['goal_obs'] = np.tile(final_data['goal_obs'], [1, ob_trajs.shape[1], 1])
                    final_data['acts'] = action_trajs

                    if self.enable_particle_based_fwd_dyn:

                        particles_trajs = data['particles_trajs'][:, :self.extract_num_time_step_data, :, :] if self.extract_num_time_step_data else data['particles_trajs']
                        particles_next_trajs = data['particles_next_trajs'][:, :self.extract_num_time_step_data, :, :] if self.extract_num_time_step_data else data['particles_next_trajs']

                        if self.downsample_idx is not None:
                            particles_trajs = particles_trajs[:, :, self.downsample_idx, :]
                            particles_next_trajs = particles_next_trajs[:, :, self.downsample_idx, :]

                        particles_trajs = np.transpose(particles_trajs, (0, 1, 3, 2))
                        if self.particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
                            # 2D Conv: reshape into [batch, time_sequence, xyz, width_cloth, height_cloth]
                            particles_trajs = np.reshape(particles_trajs, (particles_trajs.shape[0], particles_trajs.shape[1], particles_trajs.shape[2], int(np.sqrt(particles_trajs.shape[3])), int(np.sqrt(particles_trajs.shape[3]))))
                        final_data['particles'] = particles_trajs

                        particles_next_trajs = particles_next_trajs[:, -1, :, :][:, None, :, :]
                        particles_next_trajs = np.tile(particles_next_trajs, [1, particles_trajs.shape[1], 1, 1])
                        particles_next_trajs = np.transpose(particles_next_trajs, (0, 1, 3, 2))
                        if self.particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
                            # 2D Conv: reshape into [batch, time_sequence, xyz, width_cloth, height_cloth]
                            particles_next_trajs = np.reshape(particles_next_trajs, (particles_next_trajs.shape[0], particles_next_trajs.shape[1], particles_next_trajs.shape[2], int(np.sqrt(particles_next_trajs.shape[3])), int(np.sqrt(particles_next_trajs.shape[3]))))
                        final_data['goal_delta_pos_particles'] = particles_next_trajs - particles_trajs

        print('finished loading data.')
        return final_data

class DynamicsModel:

    def __init__(self, args, env_kwargs):
        if args['wandb']:
            self.wandb_run = wandb.init(
                project="deformable-soil",
                entity="ctorl",
                config=args,
                name=args.get('folder_name', ''),
            )
        else:
            self.wandb_run = None

        from envs.env import SoftGymEnvSB3
        self.env = SoftGymEnvSB3(**env_kwargs)
        self.starting_timestep = 0
        self.batch_size = args.get('batch_size', 1024)
        self.hidden_size = 32
        learning_rate = args.get('learning_rate')
        self.num_actions = args.get('num_actions')
        self.train_mode = args.get('train_mode')
        self.enable_particle_based_fwd_dyn = args.get('enable_particle_based_fwd_dyn')
        self.particle_based_fwd_dyn_impl = args.get('particle_based_fwd_dyn_impl')
        self.particle_based_cnn_lstm_fwd_dyn_mode = args.get('particle_based_cnn_lstm_fwd_dyn_mode')
        self.enable_downsampling = args.get('enable_downsampling')
        self.downsample_idx = None

        if self.particle_based_fwd_dyn_impl == 'cnn' and self.train_mode != 'fwd':
            raise NotImplementedError

        # inverse dynamics model
        self.num_action_vals_two_pickers = 6
        self.num_action_vals_one_picker = 3
        self.inv_dyn_model = InverseDynamicsModelLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], hidden_size=self.hidden_size)
        self.inv_dyn_model_optim = torch.optim.Adam(
            self.inv_dyn_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )
        self.inv_dyn_model_loss_fn = torch.nn.MSELoss()

        # forward dynamics model
        if self.enable_particle_based_fwd_dyn:
            if 'Cloth' in env_kwargs['env']:
                dimx, dimy = self.env._sample_cloth_size()[0], self.env._sample_cloth_size()[1]
                idx_p1 = 0
                idx_p2 = dimx * (dimy - 1)
                idx_p3 = dimx - 1
                idx_p4 = dimx * dimy - 1
                self.cloth_key_point_idx = np.array([idx_p1, idx_p2, idx_p3, idx_p4])
                num_particles = int(dimx * dimy)
            elif 'Rope' in env_kwargs['env']:
                num_particles = 41
            else:
                raise NotImplementedError
            self.num_particles = num_particles

            # downsample cloth
            if self.enable_downsampling:
                down_sample_scale = 4
                new_idx = np.arange(dimx * dimy).reshape((dimy, dimx))
                new_idx = new_idx[::down_sample_scale, ::down_sample_scale]
                new_cloth_ydim, new_cloth_xdim = new_idx.shape
                num_particles = new_cloth_ydim * new_cloth_xdim
                self.downsample_idx = new_idx.flatten()

            if self.particle_based_fwd_dyn_impl == 'cnn_lstm':
                self.forward_dyn_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size)
            elif self.particle_based_fwd_dyn_impl == 'cnn':
                self.forward_dyn_model = CNNForwardDynamics(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, hidden_size=self.hidden_size)
            elif self.particle_based_fwd_dyn_impl == 'lstm_reduced_obs':
                self.forward_dyn_model = ForwardDynamicsLSTMReducedObs(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, hidden_size=self.hidden_size)
            elif self.particle_based_fwd_dyn_impl == 'lstm_particles':
                self.forward_dyn_model = ForwardDynamicsLSTMParticles(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, hidden_size=self.hidden_size)
            elif self.particle_based_fwd_dyn_impl == 'perceiverio':
                self.forward_dyn_model = PerceiverIOForwardDynamics(self.env.action_space.shape[0], num_particles)
        else:
            # multi-step forward dynamics model does not work well, so we opt to use our best forward dynamics model (CNNForwardDynamicsLSTM)
            # self.forward_dyn_model = MultiStepForwardDynamicsLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], hidden_size=self.hidden_size)
            self.forward_dyn_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.observation_space.shape[0], self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size, env_kwargs=env_kwargs)
        self.forward_dyn_model_optim = torch.optim.Adam(
            self.forward_dyn_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )
        self.forward_dyn_model_loss_fn = torch.nn.MSELoss()
        if args.get('pretrain_fwd_model_ckpt'):
            checkpoint = torch.load(args.get('pretrain_fwd_model_ckpt'), map_location='cpu')
            self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
            print('Loaded pre-trained weights to forward dynamics model.')

        if args.get('pretrain_inv_model_ckpt'):
            checkpoint = torch.load(args.get('pretrain_inv_model_ckpt'), map_location='cpu')
            self.inv_dyn_model.load_state_dict(checkpoint['inverse_state_dict'])
            print('Loaded pre-trained weights to inverse dynamics model.')

        if args.get('should_freeze_fwd_model'):
            # freeze forward dynamics model's weights
            for param in self.forward_dyn_model.parameters():
                param.requires_grad = False
            print('Froze forward dynamics model!')

        # Instantiate training/validation dataset
        if args.get('is_eval') is False and args.get('visualize_fwd_model') is False:
            if self.train_mode == 'fine_tune':
                self.dataset = ExpertDemonstrations(args, self.num_action_vals_two_pickers)
            else:
                lstm_based_methods = ['cnn_lstm' , 'lstm_reduced_obs', 'lstm_particles', 'perceiverio']
                if self.particle_based_fwd_dyn_impl in lstm_based_methods:
                    self.dataset = Demonstrations(args, self.num_action_vals_one_picker, self.downsample_idx, self.particle_based_cnn_lstm_fwd_dyn_mode, env_kwargs=env_kwargs)
                elif self.particle_based_fwd_dyn_impl == 'cnn':
                    self.dataset = DemonstrationsNonLSTM(args, self.num_action_vals_one_picker, self.downsample_idx)
            self.dataset_length = len(self.dataset)
            self.train_size = int(0.8 * self.dataset_length)
            self.test_size = self.dataset_length - self.train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size])
            self.dataloader_train = DataLoader(train_dataset, batch_size=args.get('batch_size', 128), shuffle=True, num_workers=8, drop_last=True)
            self.dataloader_val = DataLoader(val_dataset, batch_size=args.get('batch_size', 128), shuffle=True, num_workers=2, drop_last=True)
        else:
            self.eval_video_path = make_dir('/'.join(args.get('checkpoint').split('/')[:-1]) + '/eval_video')

        # assign to GPU
        self.inv_dyn_model.cuda()
        self.inv_dyn_model_loss_fn.cuda()
        self.forward_dyn_model.cuda()
        self.forward_dyn_model_loss_fn.cuda()

        if self.train_mode != 'fwd':
            with open(args.get('two_arms_expert_data'), 'rb') as f:
                self.expert_data = pickle.load(f)

        print("Running dynamics model")

    def iterate_batch_inv_fwd_dyn_models(self, dataloader, is_train=False):
        total_inverse_loss = 0.0
        total_forward_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_inverse = None
            hs_forward = None

            # get batch data
            obs, goal_obs, actions = batch['ob'], batch['goal_ob'], batch['action']
            obs = obs.float().cuda()
            goal_obs = goal_obs.float().cuda()
            actions = actions.float().cuda()
            target_actions_shape = actions.shape
            target_goal_obs_shape = goal_obs.shape

            # infer predicted actions based on obs+goal_obs
            st_goal = torch.cat((obs, goal_obs), axis=2)
            batch_pred_acts, hs_inverse = self.inv_dyn_model(st_goal, hs_inverse, target_actions_shape)

            # infer predicted next_state based on predicted actions
            if self.enable_particle_based_fwd_dyn:
                particles = batch['particles'].float().cuda()
                goal_delta_pos_particles = batch['goal_delta_pos_particles'].float().cuda()

                batch_pred_delta_pos_particles, hs_forward = self.forward_dyn_model(particles, batch_pred_acts, hs_forward, goal_delta_pos_particles.shape)
            else:
                st_pred_acts = torch.cat((obs, batch_pred_acts), axis=2)

                batch_pred_next_states, hs_forward = self.forward_dyn_model(st_pred_acts, hs_forward, target_goal_obs_shape)

            # update
            if is_train:
                self.inv_dyn_model_optim.zero_grad()
                self.forward_dyn_model_optim.zero_grad()

                actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                if self.enable_particle_based_fwd_dyn:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)

                total_loss = actions_predictor_loss + states_predictor_loss
                total_loss.backward()

                self.inv_dyn_model_optim.step()
                self.forward_dyn_model_optim.step()

                total_inverse_loss += actions_predictor_loss.data.item()
                total_forward_loss += states_predictor_loss.data.item()
            else:
                actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                if self.enable_particle_based_fwd_dyn:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                total_inverse_loss += actions_predictor_loss.data.item()
                total_forward_loss += states_predictor_loss.data.item()

        return total_inverse_loss, total_forward_loss

    def iterate_batch_inv_dyn_model_fine_tune(self, dataloader, is_train=False):
        total_forward_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_inverse = None
            hs_forward = None

            # get batch data
            obs, goal_obs, actions = batch['ob'], batch['goal_ob'], batch['action']
            obs = obs.float().cuda()
            goal_obs = goal_obs.float().cuda()
            actions = actions.float().cuda()
            target_actions_shape = actions.shape
            target_goal_obs_shape = goal_obs.shape

            batch_pred_next_states = torch.cat((obs, goal_obs), axis=2)
            st_pred_acts = None
            sequence_batch_pred_delta_pos_particles = None

            if self.enable_particle_based_fwd_dyn:
                particles = batch['particles'].float().cuda()
                goal_delta_pos_particles = batch['goal_delta_pos_particles'].float().cuda()

            # predict a sequence of actions/next_states
            for _ in range(self.num_actions):
                # infer predicted actions based on obs+goal_obs
                batch_pred_acts, hs_inverse = self.inv_dyn_model(batch_pred_next_states, hs_inverse, target_actions_shape)

                # infer predicted next_state based on predicted actions
                if self.enable_particle_based_fwd_dyn:
                    batch_pred_delta_pos_particles, hs_forward = self.forward_dyn_model(particles, batch_pred_acts, hs_forward, particles.shape)
                    particles = batch_pred_delta_pos_particles + particles

                    # get key_point coordinates and manipulate the matrix into shape [batch_size, corner_index, xyz]
                    batch_pred_next_states = torch.transpose(particles[:, 0, :, self.cloth_key_point_idx], 1, 2)

                    # flatten to become [batch_size, (corner0_xyz, corner1_xyz, corner2_xyz, corner3_xyz)]
                    batch_pred_next_states = torch.flatten(batch_pred_next_states, start_dim=1)

                    # create additional dimension to match goal_obs.shape
                    batch_pred_next_states = batch_pred_next_states[:, None, :]

                    # add up all delta_pos_particles
                    if sequence_batch_pred_delta_pos_particles is None:
                        sequence_batch_pred_delta_pos_particles = batch_pred_delta_pos_particles
                    else:
                        sequence_batch_pred_delta_pos_particles = sequence_batch_pred_delta_pos_particles + batch_pred_delta_pos_particles

                else:
                    if st_pred_acts is not None:
                        # extract pred_next_states (remove goal_obs)
                        batch_pred_next_states = batch_pred_next_states[:, :, :int(batch_pred_next_states.shape[2]/2)]
                        st_pred_acts = torch.cat((batch_pred_next_states, batch_pred_acts), axis=2)
                    else:
                        st_pred_acts = torch.cat((obs, batch_pred_acts), axis=2)

                    batch_pred_next_states, hs_forward = self.forward_dyn_model(st_pred_acts, hs_forward, target_goal_obs_shape)

                batch_pred_next_states = torch.cat((batch_pred_next_states, goal_obs), axis=2)

            # update
            if is_train:
                self.inv_dyn_model_optim.zero_grad()
                self.forward_dyn_model_optim.zero_grad()
                if self.enable_particle_based_fwd_dyn:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(sequence_batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    # extract pred_next_states (remove goal_obs)
                    batch_pred_next_states = batch_pred_next_states[:, :, :int(batch_pred_next_states.shape[2]/2)]
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                states_predictor_loss.backward()
                self.inv_dyn_model_optim.step()
                self.forward_dyn_model_optim.step()
                total_forward_loss += states_predictor_loss.data.item()
            else:
                if self.enable_particle_based_fwd_dyn:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(sequence_batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    # extract pred_next_states (remove goal_obs)
                    batch_pred_next_states = batch_pred_next_states[:, :, :int(batch_pred_next_states.shape[2]/2)]
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                total_forward_loss += states_predictor_loss.data.item()

        return total_forward_loss

    def iterate_batch_inv_dyn_model(self, dataloader, enable_inv_dyn_mse_loss, is_train=False):
        total_inverse_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_inverse = None
            hs_forward = None

            # get batch data
            obs, goal_obs, actions = batch['ob'], batch['goal_ob'], batch['action']
            obs = obs.float().cuda()
            goal_obs = goal_obs.float().cuda()
            actions = actions.float().cuda()
            target_actions_shape = actions.shape
            target_goal_obs_shape = goal_obs.shape

           # infer predicted actions based on obs+goal_obs
            st_goal = torch.cat((obs, goal_obs), axis=2)
            batch_pred_acts, hs_inverse = self.inv_dyn_model(st_goal, hs_inverse, target_actions_shape)

            if self.enable_particle_based_fwd_dyn:
                particles = batch['particles'].float().cuda()
                goal_delta_pos_particles = batch['goal_delta_pos_particles'].float().cuda()

                batch_pred_delta_pos_particles, hs_forward = self.forward_dyn_model(particles, batch_pred_acts, hs_forward, goal_delta_pos_particles.shape)
            else:
                st_pred_acts = torch.cat((obs, batch_pred_acts), axis=2)
                batch_pred_next_states, hs_forward = self.forward_dyn_model(st_pred_acts, hs_forward, target_goal_obs_shape)
            # update
            if is_train:
                self.inv_dyn_model_optim.zero_grad()
                self.forward_dyn_model_optim.zero_grad()

                if self.enable_particle_based_fwd_dyn:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                if enable_inv_dyn_mse_loss:
                    actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                    total_loss = actions_predictor_loss + states_predictor_loss
                    total_inverse_loss += actions_predictor_loss.data.item()
                else:
                    total_loss = states_predictor_loss
                    total_inverse_loss += 0
                total_loss.backward()
                self.inv_dyn_model_optim.step()
                self.forward_dyn_model_optim.step()
            else:
                if enable_inv_dyn_mse_loss:
                    actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                    total_inverse_loss += actions_predictor_loss.data.item()
                else:
                    total_inverse_loss += 0

        return total_inverse_loss

    def iterate_batch_forward_dyn_model(self, dataloader, is_train=False):
        total_forward_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_forward = None

            # get batch data
            obs, goal_obs, actions = batch['ob'], batch['goal_ob'], batch['action']
            obs = obs.float().cuda()
            goal_obs = goal_obs.float().cuda()
            actions = actions.float().cuda()
            target_goal_obs_shape = goal_obs.shape

            # infer predicted next_state based on predicted actions
            if self.enable_particle_based_fwd_dyn:
                particles = batch['particles'].float().cuda()
                goal_delta_pos_particles = batch['goal_delta_pos_particles'].float().cuda()

                if self.particle_based_fwd_dyn_impl == 'lstm_reduced_obs':
                    batch_pred_next_states, hs_forward = self.forward_dyn_model(particles, actions, hs_forward, goal_obs.shape)
                elif self.particle_based_fwd_dyn_impl == 'lstm_particles' or self.particle_based_fwd_dyn_impl == 'cnn_lstm':
                    batch_pred_delta_pos_particles, hs_forward = self.forward_dyn_model(particles, actions, hs_forward, goal_delta_pos_particles.shape)
                else:
                    batch_pred_delta_pos_particles = self.forward_dyn_model(particles, actions, goal_delta_pos_particles.shape)
            else:
                batch_pred_next_states, hs_forward = self.forward_dyn_model(obs, actions, hs_forward, target_goal_obs_shape)

            # update
            if is_train:
                self.forward_dyn_model_optim.zero_grad()
                if self.enable_particle_based_fwd_dyn:
                    if self.particle_based_fwd_dyn_impl == 'lstm_reduced_obs':
                        states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)                        
                    else:
                        states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                states_predictor_loss.backward()
                self.forward_dyn_model_optim.step()
                total_forward_loss += states_predictor_loss.data.item()
            else:
                if self.enable_particle_based_fwd_dyn:
                    if self.particle_based_fwd_dyn_impl == 'lstm_reduced_obs':
                        states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                    else:
                        states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_delta_pos_particles, goal_delta_pos_particles)
                else:
                    states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, goal_obs)
                total_forward_loss += states_predictor_loss.data.item()

        return total_forward_loss

    def forward_dynamics_model_visualization(self, args, checkpoint):
        if self.enable_particle_based_fwd_dyn:
            fwd_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], self.num_particles, self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size)
        else:
            # visualization only works for particle-based forward dynamics model
            raise NotImplementedError
        fwd_model.cuda()
        fwd_model.load_state_dict(checkpoint['forward_state_dict'])
        fwd_model.eval()

        for iter in range(args.num_eval_eps):
            hs_forward = None
            self.env.reset()
            starting_config = self.env.get_current_config().copy()
            starting_state = self.env.get_state()
            sample_action = self.env.compute_random_actions()

            # get forward dynamics model's result
            self.env.start_record()

            # change to shape [1, 1, xyz, num_particles]
            particle_positions = self.env.env_pyflex.get_positions().reshape(-1, 4)[:, :3][None, None, :, :]
            particle_positions = np.transpose(particle_positions, (0, 1, 3, 2))
            particle_positions = torch.from_numpy(particle_positions).float().cuda()

            # change to shape [1, 1, pickxyz_placexyz]
            lstm_sample_action = sample_action[None, None, :]
            lstm_sample_action = torch.from_numpy(lstm_sample_action).float().cuda()

            pred_delta_particle_pos, hs_forward = fwd_model(particle_positions, lstm_sample_action, hs_forward, particle_positions.shape)

            # change from shape (1, 1, xyz, num_particles) to shape (num_particles, xyz)
            pred_delta_particle_pos = pred_delta_particle_pos.cpu().detach().numpy()[0, 0, :, :]
            pred_delta_particle_pos = np.transpose(pred_delta_particle_pos, (1, 0))

            self.env.forward_dynamics_model_rollout(pred_delta_particle_pos)
            self.env.step(np.array([-1, 2, -1, -1, 1, -1])) # useless action -> let cloth settles
            self.env.end_record(video_path=os.path.join(self.eval_video_path, f'iter_{iter}_fwd.gif'))

            # get simulator's result
            self.env.reset()
            self.env.set_scene(starting_config, starting_state)
            self.env._reset()
            self.env.start_record()
            self.env.step(sample_action)
            self.env.step(np.array([-1, 2, -1, -1, 1, -1])) # useless action -> let cloth settles
            self.env.end_record(video_path=os.path.join(self.eval_video_path, f'iter_{iter}_ground_truth.gif'))

        print('Done!')

    def evaluate(self, args, checkpoint):
        '''
        Do evaluation on the expert dataset
        '''
        model_eval = InverseDynamicsModelLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], hidden_size=self.hidden_size)
        model_eval.cuda()
        model_eval.load_state_dict(checkpoint['inverse_state_dict'])
        model_eval.eval()

        random_indices = np.random.randint(len(self.expert_data['ob_trajs']), size=args.num_eval_eps)
        total_normalized_performance_final, total_rewards, total_lengths = [], 0, 0
        action_dim = self.env.action_space.shape[0]
        for random_index in random_indices:
            self.env.reset()
            self.env.set_scene(self.expert_data['configs'][random_index], np.load(self.expert_data['state_trajs'][random_index][0], allow_pickle=True).item())
            self.env._reset()
            obs = torch.from_numpy(self.env._get_obs()).float().cuda()
            obs = obs[None, None, :-self.num_action_vals_one_picker]
            ep_len = 0
            ep_rew = 0
            ep_normalized_perf = []
            if args.eval_videos:
                self.env.start_record()
                frames = [self.env.get_image(args.eval_gif_size, args.eval_gif_size)]

            goal_obs = self.expert_data['ob_next_trajs'][random_index][0]
            goal_obs = goal_obs[None, None, :-self.num_action_vals_two_pickers]
            goal_obs = torch.from_numpy(goal_obs).float().cuda()
            st_stplus1 = torch.cat((obs, goal_obs), axis=2)

            while ep_len < self.env.horizon:
                ac_pred, _ = model_eval(st_stplus1, None, (1, action_dim))
                obs, rew, _, info = self.env.step(ac_pred[0].cpu().detach().numpy())
                obs = torch.from_numpy(obs).float().cuda()
                obs = obs[None, None, :-self.num_action_vals_one_picker]
                st_stplus1 = torch.cat((obs, goal_obs), axis=2)
                ep_len += 1
                ep_rew += rew
                ep_normalized_perf.append(info['normalized_performance'])
                if args.eval_videos:
                    frames.append(self.env.get_image(args.eval_gif_size, args.eval_gif_size))
            ep_normalized_perf_final = ep_normalized_perf[-1]
            print(f'Random index {random_index}, Episode normalized performance final: {ep_normalized_perf_final}, Rewards: {ep_rew}, Episode Length: {ep_len}')
            total_normalized_performance_final.append(ep_normalized_perf_final)
            total_rewards += ep_rew
            total_lengths += ep_len
            if args.eval_videos and ep_normalized_perf_final > 0:
                self.env.end_record(video_path=os.path.join(self.eval_video_path, f'random_index_{random_index}_{ep_normalized_perf_final}.gif'))
        del model_eval
        normalized_performance_final = np.mean(total_normalized_performance_final)
        avg_rewards = total_rewards / args.num_eval_eps
        avg_ep_length = total_lengths / args.num_eval_eps
        print(f'info_normalized_performance_final: {normalized_performance_final}')
        print(f'Average Rewards: {avg_rewards}')
        print(f'Average Episode Length: {avg_ep_length}')
        return normalized_performance_final, avg_rewards, avg_ep_length

    def save_checkpoint(self, args, epoch, filename):
        checkpoint = {
            'epoch': epoch + 1,
            'inverse_state_dict': self.inv_dyn_model.state_dict(),
            'inverse_optimizer': self.inv_dyn_model_optim.state_dict(),
            'forward_state_dict': self.forward_dyn_model.state_dict(),
            'forward_optimizer': self.forward_dyn_model_optim.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.ckpt_saved_folder, filename))
        return checkpoint

    def run(self, args):
        # Prepare for interaction with environment
        if args.resume_training:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            self.inv_dyn_model.load_state_dict(checkpoint['inverse_state_dict'])
            self.inv_dyn_model_optim.load_state_dict(checkpoint['inverse_optimizer'])
            self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
            self.forward_dyn_model_optim.load_state_dict(checkpoint['forward_optimizer'])
            self.starting_timestep = checkpoint['epoch']
            print('Loaded previously trained weights.')
            print(f'Starts training at epoch {self.starting_timestep}.')

        min_validation_forward_loss, min_validation_inverse_loss = 1000000, 1000000
        total_steps = self.starting_timestep + args.epoch
        for epoch in range(self.starting_timestep, total_steps):
            print(f'\nEpoch {epoch}')

            # process training batch data
            print('processing training batch...')
            if args.train_mode == 'fwd':
                self.forward_dyn_model.train()
                total_forward_loss = self.iterate_batch_forward_dyn_model(self.dataloader_train, is_train=True)
                training_forward_loss = total_forward_loss / (args.batch_size*len(self.dataloader_train))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Next states Performance Loss (Train): ' + str(training_forward_loss))
                print('----------------------------------------------------------------------')

                # process validation batch data
                print('\nprocessing validation batch...')
                self.forward_dyn_model.eval()
                total_forward_loss = self.iterate_batch_forward_dyn_model(self.dataloader_val, is_train=False)
                validation_forward_loss = total_forward_loss / (args.batch_size * len(self.dataloader_val))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Next states Performance Loss (Val): ' + str(validation_forward_loss))
                print('----------------------------------------------------------------------')
            elif args.train_mode == 'inv':
                self.inv_dyn_model.train()
                self.forward_dyn_model.train()
                total_inverse_loss = self.iterate_batch_inv_dyn_model(self.dataloader_train, args.enable_inv_dyn_mse_loss, is_train=True)
                training_inverse_loss = total_inverse_loss / (args.batch_size*len(self.dataloader_train))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Actions Prediction Loss (Train): ' + str(training_inverse_loss))
                print('----------------------------------------------------------------------')

                # process validation batch data
                print('\nprocessing validation batch...')
                self.inv_dyn_model.eval()
                self.forward_dyn_model.eval()
                total_inverse_loss = self.iterate_batch_inv_dyn_model(self.dataloader_val, args.enable_inv_dyn_mse_loss, is_train=False)
                validation_inverse_loss = total_inverse_loss / (args.batch_size * len(self.dataloader_val))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Actions Prediction Loss (Val): ' + str(validation_inverse_loss))
                print('----------------------------------------------------------------------')
            elif args.train_mode == 'inv_fwd':
                self.inv_dyn_model.train()
                self.forward_dyn_model.train()
                total_inverse_loss, total_forward_loss = self.iterate_batch_inv_fwd_dyn_models(self.dataloader_train, is_train=True)
                training_inverse_loss = total_inverse_loss / (args.batch_size*len(self.dataloader_train))
                training_forward_loss = total_forward_loss / (args.batch_size*len(self.dataloader_train))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Actions Prediction Loss (Train): ' + str(training_inverse_loss))
                print('Next states Performance Loss (Train): ' + str(training_forward_loss))
                print('----------------------------------------------------------------------')

                # process validation batch data
                print('\nprocessing validation batch...')
                self.inv_dyn_model.eval()
                self.forward_dyn_model.eval()
                total_inverse_loss, total_forward_loss = self.iterate_batch_inv_fwd_dyn_models(self.dataloader_val, is_train=False)
                validation_inverse_loss = total_inverse_loss / (args.batch_size * len(self.dataloader_val))
                validation_forward_loss = total_forward_loss / (args.batch_size * len(self.dataloader_val))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Actions Prediction Loss (Val): ' + str(validation_inverse_loss))
                print('Next states Performance Loss (Val): ' + str(validation_forward_loss))
                print('----------------------------------------------------------------------')
            elif args.train_mode == 'fine_tune':
                self.inv_dyn_model.train()
                self.forward_dyn_model.train()
                total_forward_loss = self.iterate_batch_inv_dyn_model_fine_tune(self.dataloader_train, is_train=True)
                training_forward_loss = total_forward_loss / (args.batch_size*len(self.dataloader_train))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Next states Performance Loss (Train): ' + str(training_forward_loss))
                print('----------------------------------------------------------------------')

                # process validation batch data
                print('\nprocessing validation batch...')
                self.inv_dyn_model.eval()
                self.forward_dyn_model.eval()
                total_forward_loss = self.iterate_batch_inv_dyn_model_fine_tune(self.dataloader_val, is_train=False)
                if total_forward_loss == 0:
                    validation_forward_loss = 0
                else:
                    validation_forward_loss = total_forward_loss / (args.batch_size * len(self.dataloader_val))
                print('\n----------------------------------------------------------------------')
                print('Epoch #' + str(epoch))
                print('Next states Performance Loss (Val): ' + str(validation_forward_loss))
                print('----------------------------------------------------------------------')

            if args.train_mode == 'fine_tune':
                if epoch % args.eval_interval == 0:
                    # arrange/save checkpoint
                    checkpoint = self.save_checkpoint(args, epoch, 'epoch_{}.pth'.format(epoch))
                    normalized_performance_final, avg_rewards, avg_ep_length = self.evaluate(args, checkpoint)
                if validation_forward_loss < min_validation_forward_loss:
                    self.save_checkpoint(args, epoch, 'lowest_fwd_loss_ckpt.pth')
                    min_validation_forward_loss = validation_forward_loss
            elif (args.train_mode == 'fwd' or args.train_mode == 'inv_fwd') and validation_forward_loss < min_validation_forward_loss:
                # lowest validation loss so far
                self.save_checkpoint(args, epoch, 'epoch_{}.pth'.format(epoch))
                min_validation_forward_loss = validation_forward_loss
            elif args.train_mode == 'inv' and validation_inverse_loss < min_validation_inverse_loss:
                # lowest validation loss so far
                self.save_checkpoint(args, epoch, 'lowest_inv_dyn_loss_epoch_{}.pth'.format(epoch))
                min_validation_inverse_loss = validation_inverse_loss

            # wandb logging
            if self.wandb_run:
                wandb_log_dict = {
                    "Epoch": epoch,
                }

                if args.train_mode == 'fwd' or args.train_mode == 'inv_fwd' or args.train_mode == 'fine_tune':
                    wandb_log_dict['Next states Performance Loss (Train)'] = training_forward_loss
                    wandb_log_dict['Next states Performance Loss (Val)'] = validation_forward_loss

                if args.train_mode == 'inv' or args.train_mode == 'inv_fwd':
                    wandb_log_dict['Actions Prediction Loss (Train)'] = training_inverse_loss
                    wandb_log_dict['Actions Prediction Loss (Val)'] = validation_inverse_loss

                if args.train_mode == 'fine_tune' and epoch % args.eval_interval == 0:
                    wandb_log_dict['val/info_normalized_performance_final'] = normalized_performance_final
                    wandb_log_dict['val/avg_rews'] = avg_rewards
                    wandb_log_dict['val/avg_ep_length'] = avg_ep_length

                wandb.log(wandb_log_dict)

        if self.wandb_run:
            self.wandb_run.finish()