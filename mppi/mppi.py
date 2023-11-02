import gym
import numpy as np
from envs.env import SoftGymEnvSB3
from inv.model_v3 import CNNForwardDynamicsLSTM
import torch
from cem_2armsto1.parallel_worker import get_cost_trained_fwd_dyn_helper, get_obs
from softgym.registered_env import env_arg_dict
import pickle
from curl import utils
import os
import datetime
import argparse
import random
from tqdm import tqdm
import wandb

reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
    'ClothFold': 50.0,
    'ClothFoldRobot': 50.0,
    'ClothFoldRobotHard': 50.0,
    'DryCloth': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': (-3, 3),
    'ClothFoldRobot': (-3, 3),
    'ClothFoldRobotHard': (-3, 3),
    'DryCloth': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}

class MPPI():
    """
    MMPI according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'
    Adapted from: https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, noise_gaussian=True, args=None, kwargs=None):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))
        self.kwargs = kwargs
        DIM_ACTION = self.kwargs['DIM_ACTION']

        self.env = env
        self.env.reset()

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T, DIM_ACTION))
        else:
            self.noise = np.full(shape=(self.K, self.T, DIM_ACTION), fill_value=0.9)

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

        pretrained_fwd_dyn_ckpt = args.pretrained_fwd_dyn_ckpt
        checkpoint = torch.load(pretrained_fwd_dyn_ckpt, map_location='cpu')
        self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
        for param in self.forward_dyn_model.parameters():
            param.requires_grad = False
        self.forward_dyn_model.cuda()
        self.forward_dyn_model.eval()

    def _get_actions_learned_fwd_dyn_model(self, action, timestep):
        # action limits from CEM
        # [pick_x, pick_y, pick_z, place_x, place_z]. We know the best place_y [0.7].

        # only for the first time step
        if timestep == 0:
            action[1] = -0.9714286 # optimal pick height for picking up cloth on table (normalized)
        else:
             # only for the second and third time steps
            action[0] = np.clip(action[0], -1.0, -0.7)
            action[1] = 0
            action[2] = np.clip(action[2], -0.6, 0.6)

        action[3] = np.clip(action[3], -1.0, -0.7)
        action[4] = np.clip(action[4], -0.6, 0.6)

        full_action = np.array([action[0], action[1], action[2], action[3], 0.7, action[4]])

        return np.array(full_action)

    def _compute_total_cost(self, k, goal_obs, extra_args, particles, hs_forward):
        current_particles = torch.clone(particles)
        if hs_forward is not None:
            hs_forward_local = (torch.clone(hs_forward[0]), torch.clone(hs_forward[1]))
        else:
            hs_forward_local = None
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            perturbed_action_t = self._get_actions_learned_fwd_dyn_model(perturbed_action_t, t)
            perturbed_action_t = torch.from_numpy(perturbed_action_t).float().cuda()

            # compute cost
            perturbed_action_t = perturbed_action_t[None, None, :]
            pred_delta_pos_particles, hs_forward_local = self.forward_dyn_model(current_particles, perturbed_action_t, hs_forward_local, current_particles.shape)
            current_particles = pred_delta_pos_particles + current_particles
            current_particles_np = current_particles.clone().cpu().detach().numpy()[0, 0, :, :]
            current_particles_np = np.transpose(current_particles_np, (1, 0))
            reward = 100 - np.linalg.norm(current_particles_np - goal_obs)
            self.cost_total[k] += -reward

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, ep_data_teacher, iter=1000):
        goal_obs = ep_data_teacher['goal_obs']
        extra_args = {
            'downsample_idx': None,
        }
        actions = []
        self.env.reset()
        self.env.set_scene(ep_data_teacher['config'], ep_data_teacher['state'])
        self.env._reset()
        particles, goal_obs = get_obs(self.env, goal_obs, 'particles', self.particle_based_cnn_lstm_fwd_dyn_mode, self.particle_based_fwd_dyn_impl, extra_args)
        hs_forward = None
        for curr_t in range(iter):
            for k in tqdm(range(self.K)):
                self._compute_total_cost(k, goal_obs, extra_args, particles, hs_forward)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            for curr_act in range(self.U.shape[1]):
                self.U[:, curr_act] += [np.sum(omega * self.noise[:, t, curr_act]) for t in range(self.T)]

            # rollout this action to get next state
            best_action = self.U[0]
            best_action = self._get_actions_learned_fwd_dyn_model(best_action, curr_t)
            best_action_torch = torch.from_numpy(best_action).float().cuda()
            best_action_torch = best_action_torch[None, None, :]
            pred_delta_pos_particles, hs_forward = self.forward_dyn_model(particles, best_action_torch, hs_forward, particles.shape)
            particles = pred_delta_pos_particles + particles

            # save the best action
            actions.append(best_action)

            # print cost
            particles_np = particles.clone().cpu().detach().numpy()[0, 0, :, :]
            particles_np = np.transpose(particles_np, (1, 0))
            reward = 100 - np.linalg.norm(particles_np - goal_obs)
            print(f'Timestep {curr_t} cost {-reward}')


            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
        return actions


    def rollout_for_stats_and_video(self, actions, ep_data_teacher, index, ep_num, out_dir, time_horizon, store_video):
        self.env.reset()
        self.env.set_scene(ep_data_teacher['config'], ep_data_teacher['state'])
        self.env._reset()

        if store_video:
            self.env.start_record()
        for curr_t in range(time_horizon):
            action = self._get_actions_learned_fwd_dyn_model(actions[curr_t], curr_t)
            obs, reward, done, info = self.env.step(action, record_continuous_video=True, img_size=256)

        normalized_perf_final = info['normalized_performance']
        if store_video:
            self.env.end_record(video_path=f'{out_dir}/index_{index}_ep_num_{ep_num}_{normalized_perf_final}.gif')
        return normalized_perf_final

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=11, type=int, help="seed number")
    parser.add_argument('--teacher_data_num_eps', default=None, type=int, help="number of episodes to compute for the teacher dataset")
    parser.add_argument('--two_arms_expert_data', default=None, type=str, help='Two arms demonstration data')
    parser.add_argument('--pretrained_fwd_dyn_ckpt', default=None, type=str, help="file path to pre-trained forward dynamics model's checkpoint")
    parser.add_argument('--wandb', action='store_true', help="whether to use wandb")
    parser.add_argument('--store_video', action='store_true', help="whether to store video")
    parser.add_argument('--mppi_timesteps', default=3, type=int, help="timesteps (hyper-parameter)")
    parser.add_argument('--mppi_num_samples', default=1000, type=int, help="number of samples (hyper-parameter)")
    args = parser.parse_args()

    DIM_ACTION = 5                             # [pick_x, pick_y, pick_z, place_x, place_z]. We know the best place_y [0.7].
    ENV_HORIZON = 3
    TIMESTEPS = args.mppi_timesteps            # T
    N_SAMPLES = args.mppi_num_samples          # K
    ACTION_LOW = -1
    ACTION_HIGH = 1

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
    'seed': args.seed,
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
    env = SoftGymEnvSB3(**config)

    kwargs = dict()
    kwargs['DIM_ACTION'] = DIM_ACTION

    # load two arms teacher dataset
    with open(args.two_arms_expert_data, 'rb') as f:
        two_arms_data = pickle.load(f)
        two_arms_data_configs = two_arms_data['configs']
        two_arms_data_state_trajs = two_arms_data['state_trajs']
        two_arms_data_goal_obs = two_arms_data['particles_next_trajs']

    utils.set_seed_everywhere(args.seed)

    if args.teacher_data_num_eps is not None:
        if args.teacher_data_num_eps == len(two_arms_data_configs):
            # iterate all episodes in the two arms teacher dataset
            indices = [i for i in range(len(two_arms_data_configs))]
        else:
            lst = range(0, len(two_arms_data_configs))
            indices = random.choices(lst, k=args.teacher_data_num_eps) # no repeat random numbers between 0 and len(two_arms_data)
    else:
        indices = [0]

    # set up output directory
    timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')
    run_name = f"MPPI_{timestamp}"
    outdir = os.path.join(os.getcwd(), "data", "mppi", run_name)
    os.makedirs(outdir)

    if args.wandb:
        wandb_run = wandb.init(
            project="deformable-soil",
            entity="ctorl",
            config=config,
            name=run_name,
        )
    else:
        wandb_run = None

    print('mppi_timesteps: ',  args.mppi_timesteps)
    print('mppi_num_samples: ', args.mppi_num_samples)
    avg_normalized_perf_final, action_trajs, fitness_trajs = [], [], []
    for index, ep_num in enumerate(indices):
        print('teacher dataset episode ' + str(ep_num))

        # construct episode data from teacher dataset
        ep_data_teacher = {
            'config': two_arms_data_configs[ep_num],
            'state': two_arms_data_state_trajs[ep_num, 0],
            'goal_obs': two_arms_data_goal_obs[ep_num, 0],
        }

        # get action using MPPI
        noise_mu = 0
        noise_sigma = 10
        lambda_ = 1
        U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=(TIMESTEPS, DIM_ACTION))
        mppi_gym = MPPI(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True, args=args, kwargs=kwargs)
        actions = mppi_gym.control(ep_data_teacher, iter=ENV_HORIZON)
        action_trajs.append(actions)
        normalized_perf_final = mppi_gym.rollout_for_stats_and_video(actions, ep_data_teacher, index, ep_num, outdir, ENV_HORIZON, args.store_video)
        if wandb_run:
            wandb_log_dict = {
                "Episode": index,
                "val/info_normalized_performance_final": normalized_perf_final,
            }
            wandb.log(wandb_log_dict)
        else:
            print('info_normalized_performance_final: ', normalized_perf_final)
        avg_normalized_perf_final.append(normalized_perf_final)

    traj_dict = {
        'action_trajs': np.array(action_trajs),
        'two_arms_data_indices': np.array(indices),
        'total_normalized_perf_final_first_100eps': np.array(avg_normalized_perf_final),
    }
    with open(os.path.join(outdir, f'mppi_traj.pkl'), 'wb') as file_handle:
        pickle.dump(traj_dict, file_handle)

    avg_normalized_perf_final = np.average(avg_normalized_perf_final)
    if wandb_run:
        wandb_log_dict = {
            "val/avg_info_normalized_performance_final": avg_normalized_perf_final,
        }
        wandb.log(wandb_log_dict)
        wandb_run.finish()
    else:
        print(f'val/avg_info_normalized_performance_final: {avg_normalized_perf_final}')