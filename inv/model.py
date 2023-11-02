import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from envs.env import SoftGymEnvSB3
from torch.utils.data import DataLoader, Dataset
import pickle
import tqdm
import os

class InvDynMLP(nn.Module):
    """MLP inverse dynamics model."""

    def __init__(self, obs_dim, act_dim, num_actions, mlp_w=32):
        super(InvDynMLP, self).__init__()
        # Build the model
        self.fc0 = nn.Linear(obs_dim * 2, mlp_w)
        self.fc1 = nn.Linear(mlp_w, mlp_w)
        self.fc2 = nn.Linear(mlp_w, num_actions * act_dim)

    def forward(self, x):
        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class ActionsToStatesMLP(nn.Module):
    """MLP actions to states mapping."""

    def __init__(self, obs_dim, act_dim, num_actions, mlp_w=32):
        super(ActionsToStatesMLP, self).__init__()
        # Build the model
        self.fc0 = nn.Linear(num_actions * act_dim, mlp_w)
        self.fc1 = nn.Linear(mlp_w, mlp_w)
        self.fc2 = nn.Linear(mlp_w, obs_dim)

    def forward(self, x):
        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class InverseDynamicsPolicy(nn.Module):

    def __init__(self, obs_dim, act_dim, num_actions, mlp_w=32):
        super(InverseDynamicsPolicy, self).__init__()
        self.inv_dyn_mlp = InvDynMLP(obs_dim, act_dim, num_actions)
        self.act_state_mlp = ActionsToStatesMLP(obs_dim, act_dim, num_actions)

    def forward(self, x):
        pred_acts = self.inv_dyn_mlp(x)
        pred_next_states = self.act_state_mlp(pred_acts)
        return pred_acts, pred_next_states

class Demonstrations(Dataset):
    def __init__(self, args, num_action_vals_two_pickers):
        self.num_action_vals_two_pickers = num_action_vals_two_pickers
        self.data = self.load_file(args['two_arms_expert_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, next_ob, config, state_traj  = self.data["obs"][idx], self.data["next_obs"][idx], self.data["configs"][idx],  self.data["state_trajs"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'next_ob': torch.from_numpy(next_ob),
            'state_traj': state_traj,
            'ClothPos': np.array(config['ClothPos']),
            'ClothSize': np.array(config['ClothSize']),
            'ClothStiff': np.array(config['ClothStiff']),
            'camera_name': config['camera_name'],
            'pos': np.array(config['camera_params']['default_camera']['pos']),
            'angle': np.array(config['camera_params']['default_camera']['angle']),
            'width': config['camera_params']['default_camera']['width'],
            'height': config['camera_params']['default_camera']['height'],
            'flip_mesh': config['flip_mesh']
        }
        return out

    def load_file(self, file_path):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'next_obs': [],
            'configs': [],
            'state_trajs': [],
        }
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            ob_trajs = data['ob_trajs']
            next_obs_trajs = data['ob_next_trajs']
            configs = data['configs']
            state_trajs = data['state_trajs']

            for obs_ep, next_obs_ep, config, state_traj in zip(ob_trajs, next_obs_trajs, configs, state_trajs):
                for ob, next_ob, state in zip(obs_ep, next_obs_ep, state_traj):
                    final_data['obs'].append(ob[:-self.num_action_vals_two_pickers])
                    final_data['next_obs'].append(next_ob[:-self.num_action_vals_two_pickers])
                    final_data['configs'].append(config)
                    final_data['state_trajs'].append(state)
            final_data['obs'] = np.array(final_data['obs'])
            final_data['next_obs'] = np.array(final_data['next_obs'])
            final_data['configs'] = np.array(final_data['configs'])
            final_data['state_trajs'] = np.array(final_data['state_trajs'])
        print('finished loading data.')
        return final_data

class InverseDynamicsModel:

    def __init__(self, args, env_kwargs):
        if args['wandb']:
            self.wandb_run = wandb.init(
                project="cto-rl-manipulation",
                config=args,
                name=args.get('folder_name', ''),
            )
        else:
            self.wandb_run = None

        self.env = SoftGymEnvSB3(**env_kwargs)
        self.starting_timestep = 0
        self.batch_size = args.get('batch_size', 1024)

        # inverse dynamics model
        self.num_action_vals_two_pickers = 6
        self.num_action_vals_one_picker = 3
        self.num_actions = args.get('num_actions')
        self.inv_dyn_model = InverseDynamicsPolicy(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], self.num_actions)
        self.inv_dyn_model_optim = torch.optim.Adam(
            self.inv_dyn_model.parameters(),
            lr=1e-3,
            weight_decay=0.0,
        )
        self.inv_dyn_model_loss_fn = torch.nn.MSELoss()

        # Instantiate training/validation dataset
        self.dataset = Demonstrations(args, self.num_action_vals_two_pickers)
        self.dataset_length = len(self.dataset)
        self.train_size = int(0.8 * self.dataset_length)
        self.test_size = self.dataset_length - self.train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.test_size])
        self.dataloader_train = DataLoader(train_dataset, batch_size=args.get('batch_size', 128), shuffle=True, num_workers=2, drop_last=True)
        self.dataloader_val = DataLoader(val_dataset, batch_size=args.get('batch_size', 128), shuffle=True, num_workers=2, drop_last=True)

        # assign to GPU
        self.inv_dyn_model.cuda()
        self.inv_dyn_model_loss_fn.cuda()

        print("Running Inverse Dynamics Model")

    def has_image_observations(self):
        return self.env.observation_mode in ['cam_rgb_key_point', 'depth_key_point']

    def construct_configs(self, batch):
        '''
        Convert batch configs to this config structure:
            {'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [80, 80],
            'ClothStiff': [0.8, 1, 0.9],
            'camera_name': 'default_camera',
            'camera_params': {
                'default_camera': {
                    'pos': array([-0.,  1.,  0.]),
                    'angle': array([ 0.        , -1.57079633,  0.        ]),
                    'width': 720,
                    'height': 720
                }
            },
            'flip_mesh': 1}
        '''
        configs = []
        for ClothPos, ClothSize, ClothStiff, camera_name, pos, angle, width, height, flip_mesh in zip(batch['ClothPos'], batch['ClothSize'], batch['ClothStiff'], batch['camera_name'], batch['pos'], batch['angle'], batch['width'], batch['height'], batch['flip_mesh']):
            curr_dict = dict()
            curr_dict['ClothPos'] = ClothPos.cpu().detach().numpy()
            curr_dict['ClothSize'] = ClothSize.cpu().detach().numpy()
            curr_dict['ClothStiff'] = ClothStiff.cpu().detach().numpy()
            curr_dict['camera_name'] = camera_name
            curr_dict['camera_params'] = {
                'default_camera': {
                    'pos': pos.cpu().detach().numpy(),
                    'angle': angle.cpu().detach().numpy(),
                    'width': width.cpu().detach().item(),
                    'height': height.cpu().detach().item(),
                }
            }
            curr_dict['flip_mesh'] = flip_mesh.cpu().detach().item()
            configs.append(curr_dict)
        return configs

    def iterate_batch_data_update_model(self, dataloader, is_train=False):
        total_loss = 0.0
        total_rewards = 0.0
        batch_tqdm = tqdm.tqdm(total=len(list(enumerate(dataloader))), desc='Batch Data', position=0)
        for _, batch in enumerate(dataloader):

            # get batch data
            obs, next_obs, configs, state_trajs = batch['ob'], batch['next_ob'], self.construct_configs(batch), batch['state_traj']
            obs = obs.float().cuda()
            next_obs = next_obs.float().cuda()

            # infer predicted actions and next_states based on obs+next_obs
            st_stplus1 = torch.cat((obs, next_obs), axis=1)
            batch_pred_acts, batch_pred_next_states = self.inv_dyn_model(st_stplus1)
            batch_pred_acts_no_grads = batch_pred_acts.clone().detach().cpu().numpy()

            # construct gt_env_next_obs using our model's predicted actions
            gt_env_next_obs = []
            norm_perf_final = []
            for pred_acts, config, state_traj in zip(batch_pred_acts_no_grads, configs, state_trajs):
                pred_acts = pred_acts.reshape(int(pred_acts.shape[0] / self.env.action_space.shape[0]), self.env.action_space.shape[0])
                self.env.reset()
                self.env.set_scene(config, np.load(state_traj, allow_pickle=True).item())
                self.env._reset()
                for pred_act in pred_acts:
                    gt_next_ob, rew, done, info = self.env.step(pred_act)
                gt_env_next_obs.append(gt_next_ob[:-self.num_action_vals_one_picker])
                norm_perf_final.append(rew)
            gt_env_next_obs = torch.tensor(gt_env_next_obs).float().cuda()

            if is_train:
                # minimize mse error between InvDynModel's predicted s_t+1 and environment s_t+1 using our model's predicted actions AND
                # minimize mse error between InvDynModel's predicted s_t+1 and the expert demonstrations' s_t+1
                state_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_next_states, gt_env_next_obs) * 0.5 + self.inv_dyn_model_loss_fn(batch_pred_next_states, next_obs) * 0.5
                self.inv_dyn_model_optim.zero_grad()
                state_predictor_loss.backward()
                self.inv_dyn_model_optim.step()
            else:
                state_predictor_loss = torch.nn.functional.mse_loss(gt_env_next_obs, next_obs)

            # update loss and rewards
            total_loss += state_predictor_loss.data.item()
            total_rewards += np.mean(norm_perf_final)

            # update tqdm counter
            batch_tqdm.update(1)
        return total_loss, total_rewards

    def run(self, args):
        # Prepare for interaction with environment
        total_steps = self.starting_timestep + args.epoch

        for epoch in range(self.starting_timestep, total_steps):
            print(f'\nEpoch {epoch}')

            # process training batch data
            print('processing training batch...')
            self.inv_dyn_model.train()
            total_loss, total_rewards = self.iterate_batch_data_update_model(self.dataloader_train, is_train=True)
            training_loss = total_loss / (args.batch_size*len(self.dataloader_train))
            train_mean_norm_per_final = total_rewards / (args.batch_size*len(self.dataloader_train))
            print('\n----------------------------------------------------------------------')
            print('Epoch #' + str(epoch))
            print('State Prediction Loss (Train): ' + str(training_loss))
            print('Mean normalized performance final (Train): ' + str(train_mean_norm_per_final))
            print('----------------------------------------------------------------------')


            # process validation batch data
            print('\nprocessing validation batch...')
            self.inv_dyn_model.eval()
            total_loss, total_rewards = self.iterate_batch_data_update_model(self.dataloader_val, is_train=False)
            validation_loss = total_loss / (args.batch_size * len(self.dataloader_val))
            val_mean_norm_per_final = total_rewards / (args.batch_size*len(self.dataloader_val))
            print('\n----------------------------------------------------------------------')
            print('Epoch #' + str(epoch))
            print('State Prediction Loss (Val): ' + str(validation_loss))
            print('Mean normalized performance final (Val): ' + str(val_mean_norm_per_final))
            print('----------------------------------------------------------------------')

            # arrange/save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.inv_dyn_model.state_dict(),
                'optimizer': self.inv_dyn_model_optim.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.ckpt_saved_folder, 'epoch_{}.pth'.format(epoch)))

            # wandb logging
            if self.wandb_run:
                wandb.log({
                    "Epoch": epoch,
                    "State Prediction Loss (Train):": training_loss,
                    "Mean normalized performance final (Train)": train_mean_norm_per_final,
                    "State Prediction Loss (Val):": validation_loss,
                    "Mean normalized performance final (Val)": val_mean_norm_per_final,
                })

        if self.wandb_run:
            self.wandb_run.finish()