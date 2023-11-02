import os
from optim_prob import optim_prob
import pyflex

import matplotlib.pyplot as plt
import matplotlib # avoid type3 fonts for papers
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from envs.env import SoftGymEnvSB3
import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import Rbf
from misc import reward_scales, clip_obs
from softgym.registered_env import env_arg_dict
from inv.model_v3 import CNNForwardDynamicsLSTM
import torch
from cem_2armsto1.parallel_worker import get_cost_trained_fwd_dyn_helper

class DryClothProb(optim_prob):
    def __init__(self, kwargs):
        # store arguments
        self.enable_trained_fwd_dyn = kwargs.get('enable_trained_fwd_dyn')

        # prepare environment arguments & instantiate environment
        env_name = 'DryCloth'
        env_kwargs = env_arg_dict[env_name]
        env_kwargs['observation_mode'] = 'key_point'
        env_kwargs['action_mode'] = 'pickerpickandplace'
        env_kwargs['num_picker'] = 1
        env_kwargs['action_repeat'] = 1
        env_kwargs['num_variations'] = 100
        env_kwargs['horizon'] = 3
        obs_mode = env_kwargs['observation_mode']
        not_imaged_based = env_kwargs['observation_mode'] not in ['cam_rgb', 'cam_rgb_key_point', 'depth_key_point']
        symbolic = not_imaged_based
        scale_reward = reward_scales[env_name]
        clipping = clip_obs[env_name] if obs_mode == 'key_point' else None
        config = {
        'env': 'DryCloth',
        'symbolic': symbolic,
        'seed': kwargs.get('seed'),
        'max_episode_length': 10,
        'action_repeat': 1,
        'bit_depth': 8,
        'image_dim': None if not_imaged_based else env_kwargs['env_image_size'],
        'env_kwargs': env_kwargs,
        'normalize_observation': False,
        'scale_reward': scale_reward,
        'clip_obs': clipping,
        'obs_process': None,
        }
        config['env_kwargs'] = env_kwargs
        print('Env config: ', config)
        self.env = SoftGymEnvSB3(**config)

        if self.enable_trained_fwd_dyn:
            self._dim_action = 5 # [pick_x, pick_y, pick_z, place_x, place_z]. We know the best place_y [0.7].
            
            # learned forward dynamics model
            self.num_action_vals_one_picker = 3
            self.num_action_vals_two_pickers = 6
            self.hidden_size = 32
            self.particle_based_cnn_lstm_fwd_dyn_mode = '1dconv'
            self.particle_based_fwd_dyn_impl = 'cnn_lstm'
            self.lstm_based_methods = ['cnn_lstm' , 'lstm_particles']
            cloth_dim_xandy = self.env._sample_cloth_size()[0]
            num_particles = int(cloth_dim_xandy * cloth_dim_xandy)
            self.forward_dyn_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0] - self.num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size)

            pretrained_fwd_dyn_ckpt = kwargs.get('pretrained_fwd_dyn_ckpt')
            checkpoint = torch.load(pretrained_fwd_dyn_ckpt, map_location='cpu')
            self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
            for param in self.forward_dyn_model.parameters():
                param.requires_grad = False
            self.forward_dyn_model.cuda()
            self.forward_dyn_model.eval()
        else:
            self._dim_action = 4 # [cornerpickIDX, placex, placey, placez] self.env.action_space.shape[0]
        
        self._nsteps = self.env.horizon
        super(DryClothProb, self).__init__()

    def fitness(self, actions, ep_data_teacher):
        if self.enable_trained_fwd_dyn:
            total_cost = self.rollout(actions, ep_data_teacher)
            return total_cost
        else:
            self.rollout(actions, ep_data_teacher)
            info = self.env._get_info()

            # -reward because optimizer wants a cost function
            return -info['normalized_performance']

    def fitness_directTO(self, dv):
         # no direct trajopt implemented
        raise NotImplementedError

    def fitness_indirectTO(self, actions, ep_data_teacher=None):
        if ep_data_teacher is not None:
            # initial guess, save this for future use during this episode's optimization
            self.ep_data_teacher = ep_data_teacher
        return self.fitness(actions, self.ep_data_teacher)

    def action_with_pick_index(self, action):
        pickidx = int(action[0] > 0) # +ve => corner1, -ve => corner0
        pos = pyflex.get_positions().reshape(-1, 4)
        corner = pos[self.env._wrapped_env.corner_pick_idx[pickidx]][0:3] # x,y,z
        pick_action = self.env._wrapped_env.normalize_action_pick_place(corner)[0:3]
        place_action = action[1:4]
        full_action = np.concatenate([pick_action, place_action])
        return full_action

    def get_actions_learned_fwd_dyn_model(self, actions):
        new_actions = []

        # action limits from CEM
        # [pick_x, pick_y, pick_z, place_x, place_z]. We know the best place_y [0.7].

        # only for the first time step
        actions[0, 1] = -0.9714286 # optimal pick height for picking up cloth on table (normalized)

        # only for the second and third time steps
        actions[1:, 0] = np.clip(actions[1:, 0], -1.0, -0.7) 
        actions[1:, 1] = 0
        actions[1:, 2] = np.clip(actions[1:, 2], -0.6, 0.6)
 
        actions[:, 3] = np.clip(actions[:, 3], -1.0, -0.7) 
        actions[:, 4] = np.clip(actions[:, 4], -0.6, 0.6) 

        for action in actions:
            full_action = np.array([action[0],  action[1], action[2], action[3], 0.7, action[4]])
            new_actions.append(full_action)

        return np.array(new_actions)

    def rollout(self, actions, ep_data_teacher, video=False):
        assert actions.size == self._nsteps * self._dim_action
        actions = np.array(actions).reshape(-1, self._dim_action)

        self.env.reset()
        
        if self.enable_trained_fwd_dyn:
            self.env.set_scene(ep_data_teacher['config'], ep_data_teacher['state'])
            self.env._reset()
            goal_obs = ep_data_teacher['goal_obs']

            extra_args = {
                'downsample_idx': None,
            }
            actions = self.get_actions_learned_fwd_dyn_model(actions)
            actions = torch.from_numpy(actions).float().cuda()
            total_cost = get_cost_trained_fwd_dyn_helper(self.env, actions, self.particle_based_fwd_dyn_impl, self.lstm_based_methods, 'particles', self.forward_dyn_model, goal_obs, \
                self.num_action_vals_two_pickers, self.particle_based_cnn_lstm_fwd_dyn_mode, extra_args)
            return total_cost
        else:
            if video:
                self.env.start_record()

            for action in actions:
                full_action = self.action_with_pick_index(action)
                self.env.step(full_action, record_continuous_video=video, img_size=128)

            if video:
                self.env.end_record(video_path=f'./data/DryCloth.gif')
                print(f'Saved video!')
        return

    def rollout_for_stats_and_video(self, actions, ep_data_teacher, index, ep_num, out_dir):
        self.env.reset()
        self.env.set_scene(ep_data_teacher['config'], ep_data_teacher['state'])
        self.env._reset()

        self.env.start_record()
        actions = actions.reshape(-1, self._dim_action)
        actions = self.get_actions_learned_fwd_dyn_model(actions)
        for action in actions:
            obs, reward, done, info = self.env.step(action, record_continuous_video=True, img_size=256)

        normalized_perf_final = info['normalized_performance']
        self.env.end_record(video_path=f'{out_dir}/index_{index}_ep_num_{ep_num}_{normalized_perf_final}.gif')
        return normalized_perf_final

    def get_name(self):
        return 'DryCloth'
