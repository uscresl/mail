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
from softgym.utils.visualization import save_numpy_as_gif
from sb3.utils import make_dir

class InverseDynamicsModelLSTM(nn.Module):
    """LSTM implementation"""

    def __init__(self, obs_dim, act_dim, hidden_size=32, num_layers=1):
        super(InverseDynamicsModelLSTM, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(obs_dim * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, act_dim)


    def forward(self, x, hs, target_shape):
        out, hs = self.lstm(x, hs)              # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)     
        out = self.fc(out)                      # out.shape = (batch_size * seq_len, act_dim)
        out = out.reshape(target_shape)
        return out, hs

class MultiStepForwardDynamicsLSTM(nn.Module):
    """LSTM implementation"""

    def __init__(self, obs_dim, act_dim, hidden_size=32, num_layers=1):
        super(MultiStepForwardDynamicsLSTM, self).__init__()
        # Build the model
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(act_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, obs_dim)

    def forward(self, x, hs, target_shape):
        out, hs = self.lstm(x, hs)               # out.shape = (batch_size, seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size) # out.shape = (batch_size * seq_len, hidden_size)     
        out = self.fc(out)                      # out.shape = (batch_size * seq_len, obs_dim) 
        out = out.reshape(target_shape)
        return out, hs

class ExpertDemonstrations(Dataset):
    def __init__(self, args, num_action_vals_two_pickers):
        self.num_action_vals_two_pickers = num_action_vals_two_pickers
        self.data = self.load_file(args['two_arms_expert_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, next_ob, action = self.data["obs"][idx], self.data["next_obs"][idx], self.data["acts"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'next_ob': torch.from_numpy(next_ob),
            'action': torch.from_numpy(action),
        }
        return out

    def load_file(self, file_path):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'next_obs': [],
            'acts': [],
        }
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            ob_trajs = data['ob_trajs']
            next_obs_trajs = data['ob_next_trajs']
            action_trajs = data['action_trajs']

            final_data['obs'] = ob_trajs[:, 0, :-self.num_action_vals_two_pickers]
            final_data['next_obs'] = next_obs_trajs[:, 0, :-self.num_action_vals_two_pickers]
            final_data['acts'] = action_trajs[:, 0]
            final_data['obs'] = final_data['obs'][:, None, :]
            final_data['next_obs'] = final_data['next_obs'][:, None, :]
            final_data['acts'] = final_data['acts'][:, None, :]
        print('finished loading data.')
        return final_data

class Demonstrations(Dataset):
    def __init__(self, args, num_action_vals_one_picker):
        self.num_action_vals_one_picker = num_action_vals_one_picker
        self.data = self.load_file(args['random_actions_data'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, next_ob, action  = self.data["obs"][idx], self.data["next_obs"][idx], self.data["acts"][idx]

        out = {
            'ob': torch.from_numpy(ob),
            'next_ob': torch.from_numpy(next_ob),
            'action': torch.from_numpy(action),
        }
        return out

    def load_file(self, file_paths):
        print('loading all data to RAM before training....')

        final_data = {
            'obs': [],
            'next_obs': [],
            'acts': [],
        }
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                ob_trajs = data['ob_trajs']
                next_obs_trajs = data['ob_next_trajs']
                action_trajs = data['action_trajs']

                if len(final_data['obs']) == 0:
                    final_data['obs'] = ob_trajs[:, :, :-self.num_action_vals_one_picker]
                    final_data['next_obs'] = next_obs_trajs[:, :, :-self.num_action_vals_one_picker]
                    final_data['acts'] = action_trajs
                else:
                    final_data['obs'] = np.concatenate((final_data['obs'], ob_trajs[:, :, :-self.num_action_vals_one_picker]), axis=0)
                    final_data['next_obs'] = np.concatenate((final_data['next_obs'], next_obs_trajs[:, :, :-self.num_action_vals_one_picker]), axis=0)
                    final_data['acts'] = np.concatenate((final_data['acts'], action_trajs), axis=0)
        print('finished loading data.')
        return final_data

class InverseDynamicsModel:

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

        self.env = SoftGymEnvSB3(**env_kwargs)
        self.starting_timestep = 0
        self.batch_size = args.get('batch_size', 1024)
        self.hidden_size = 32
        learning_rate = args.get('learning_rate')
        self.num_actions = args.get('num_actions')

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
        self.forward_dyn_model = MultiStepForwardDynamicsLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], hidden_size=self.hidden_size)
        self.forward_dyn_model_optim = torch.optim.Adam(
            self.forward_dyn_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0,
        )
        self.forward_dyn_model_loss_fn = torch.nn.MSELoss()

        # Instantiate training/validation dataset
        if args.get('is_eval') is False:
            if args.get('enable_fine_tuning'):
                self.dataset = ExpertDemonstrations(args, self.num_action_vals_two_pickers)
            else:
                self.dataset = Demonstrations(args, self.num_action_vals_one_picker)
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

        with open(args.get('two_arms_expert_data'), 'rb') as f:
            self.expert_data = pickle.load(f)

        print("Running Inverse Dynamics Model")

    def iterate_batch_data_update_model_fine_tuning(self, dataloader, is_train=False):
        total_inverse_loss = 0.0
        total_forward_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_inverse = None
            hs_forward = None

            # get batch data
            obs, next_obs, actions = batch['ob'], batch['next_ob'], batch['action']
            obs = obs.float().cuda()
            next_obs = next_obs.float().cuda()
            actions = actions.float().cuda()
            target_actions_shape = list(actions.shape)
            target_actions_shape[2] = int(target_actions_shape[2] / 2) # expert actions have 2 actuators, where here we only want one actuator
            target_next_obs_shape = next_obs.shape

            batch_pred_next_states = torch.cat((obs, next_obs), axis=2)
            sequence_batch_pred_next_states = None
            # predict a sequence of actions/next_states
            for _ in range(self.num_actions):
                # infer predicted actions based on obs+next_obs
                batch_pred_acts, hs_inverse = self.inv_dyn_model(batch_pred_next_states, hs_inverse, target_actions_shape)

                # infer predicted next_state based on predicted actions
                batch_pred_next_states, hs_forward = self.forward_dyn_model(batch_pred_acts, hs_forward, target_next_obs_shape)

                # accumulate next_states, so they're up to self.num_actions size
                if sequence_batch_pred_next_states is None:
                    sequence_batch_pred_next_states = batch_pred_next_states
                else:
                    sequence_batch_pred_next_states = torch.cat((sequence_batch_pred_next_states, batch_pred_next_states), axis=1)
                # detach from computation graph as we need to append expert's next_obs to our current obs
                batch_pred_next_states = batch_pred_next_states.clone().detach()
                batch_pred_next_states = torch.cat((batch_pred_next_states, next_obs), axis=2)

            # (batch_size, 1, obs_size) to (batch_size, self.num_actions, obs_size)
            next_obs = next_obs.repeat(1, self.num_actions, 1) # https://stackoverflow.com/questions/57896357/how-to-repeat-tensor-in-a-specific-new-dimension-in-pytorch

            # update
            if is_train:
                self.inv_dyn_model_optim.zero_grad()
                self.forward_dyn_model_optim.zero_grad()

                states_predictor_loss = self.forward_dyn_model_loss_fn(sequence_batch_pred_next_states, next_obs)
                
                total_loss = states_predictor_loss
                total_loss.backward()

                self.inv_dyn_model_optim.step()
                self.forward_dyn_model_optim.step()

                total_inverse_loss = 0
                total_forward_loss += states_predictor_loss.data.item()
            else:      
                states_predictor_loss = self.forward_dyn_model_loss_fn(sequence_batch_pred_next_states, next_obs)
                total_inverse_loss = 0
                total_forward_loss += states_predictor_loss.data.item()

        return total_inverse_loss, total_forward_loss

    def iterate_batch_data_update_model(self, dataloader, is_train=False):
        total_inverse_loss = 0.0
        total_forward_loss = 0.0
        for _, batch in enumerate(dataloader):
            hs_inverse = None
            hs_forward = None

            # get batch data
            obs, next_obs, actions = batch['ob'], batch['next_ob'], batch['action']
            obs = obs.float().cuda()
            next_obs = next_obs.float().cuda()
            actions = actions.float().cuda()
            target_actions_shape = actions.shape
            target_next_obs_shape = next_obs.shape

            # infer predicted actions based on obs+next_obs
            st_stplus1 = torch.cat((obs, next_obs), axis=2)
            batch_pred_acts, hs_inverse = self.inv_dyn_model(st_stplus1, hs_inverse, target_actions_shape)
            # batch_pred_acts_no_grads = batch_pred_acts.clone().detach() # detached pred_acts

            # infer predicted next_state based on predicted actions
            batch_pred_next_states, hs_forward = self.forward_dyn_model(batch_pred_acts, hs_forward, target_next_obs_shape)

            # update
            if is_train:
                self.inv_dyn_model_optim.zero_grad()
                self.forward_dyn_model_optim.zero_grad()

                actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, next_obs)
                
                total_loss = actions_predictor_loss + states_predictor_loss
                total_loss.backward()

                self.inv_dyn_model_optim.step()
                self.forward_dyn_model_optim.step()

                total_inverse_loss += actions_predictor_loss.data.item()
                total_forward_loss += states_predictor_loss.data.item()
            else:      
                actions_predictor_loss = self.inv_dyn_model_loss_fn(batch_pred_acts, actions)
                states_predictor_loss = self.forward_dyn_model_loss_fn(batch_pred_next_states, next_obs)
                total_inverse_loss += actions_predictor_loss.data.item()
                total_forward_loss += states_predictor_loss.data.item()

        return total_inverse_loss, total_forward_loss

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

            target_next_obs = self.expert_data['ob_next_trajs'][random_index][0]
            target_next_obs = target_next_obs[None, None, :-self.num_action_vals_two_pickers]
            target_next_obs = torch.from_numpy(target_next_obs).float().cuda()
            st_stplus1 = torch.cat((obs, target_next_obs), axis=2)

            while ep_len < self.env.horizon:
                ac_pred, _ = model_eval(st_stplus1, None, (1, action_dim))
                obs, rew, _, info = self.env.step(ac_pred[0].cpu().detach().numpy())
                obs = torch.from_numpy(obs).float().cuda()
                obs = obs[None, None, :-self.num_action_vals_one_picker]
                st_stplus1 = torch.cat((obs, target_next_obs), axis=2)
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

        if args.enable_fine_tuning:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            self.inv_dyn_model.load_state_dict(checkpoint['inverse_state_dict'])
            self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
            print('Loaded pretrained weights from training with random actions dataset.')

        total_steps = self.starting_timestep + args.epoch
        for epoch in range(self.starting_timestep, total_steps):
            print(f'\nEpoch {epoch}')

            # process training batch data
            print('processing training batch...')
            self.inv_dyn_model.train()
            self.forward_dyn_model.train()
            if args.enable_fine_tuning:
                total_inverse_loss, total_forward_loss = self.iterate_batch_data_update_model_fine_tuning(self.dataloader_train, is_train=True)
            else:
                total_inverse_loss, total_forward_loss = self.iterate_batch_data_update_model(self.dataloader_train, is_train=True)
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
            if args.enable_fine_tuning:
                total_inverse_loss, total_forward_loss = self.iterate_batch_data_update_model_fine_tuning(self.dataloader_val, is_train=False)
            else:
                total_inverse_loss, total_forward_loss = self.iterate_batch_data_update_model(self.dataloader_val, is_train=False)
            validation_inverse_loss = total_inverse_loss / (args.batch_size * len(self.dataloader_val))
            validation_forward_loss = total_forward_loss / (args.batch_size * len(self.dataloader_val))
            print('\n----------------------------------------------------------------------')
            print('Epoch #' + str(epoch))
            print('Actions Prediction Loss (Val): ' + str(validation_inverse_loss))
            print('Next states Performance Loss (Val): ' + str(validation_forward_loss))
            print('----------------------------------------------------------------------')

            if epoch % args.eval_interval == 0:
                # arrange/save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'inverse_state_dict': self.inv_dyn_model.state_dict(),
                    'inverse_optimizer': self.inv_dyn_model_optim.state_dict(),
                    'forward_state_dict': self.forward_dyn_model.state_dict(),
                    'forward_optimizer': self.forward_dyn_model_optim.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.ckpt_saved_folder, 'epoch_{}.pth'.format(epoch)))

                normalized_performance_final, avg_rewards, avg_ep_length = self.evaluate(args, checkpoint)

            # wandb logging
            if self.wandb_run:
                wandb_log_dict = {
                    "Epoch": epoch,
                    "Actions Prediction Loss (Train)": training_inverse_loss,
                    "Next states Performance Loss (Train)": training_forward_loss,
                    "Actions Prediction Loss (Val)": validation_inverse_loss,
                    "Next states Performance Loss (Val)": validation_forward_loss,
                }

                if epoch % args.eval_interval == 0:
                    wandb_log_dict['val/info_normalized_performance_final'] = normalized_performance_final
                    wandb_log_dict['val/avg_rews'] = avg_rewards
                    wandb_log_dict['val/avg_ep_length'] = avg_ep_length

                wandb.log(wandb_log_dict)

        if self.wandb_run:
            self.wandb_run.finish()