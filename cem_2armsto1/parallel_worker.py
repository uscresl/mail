import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from inv.model_v3 import MultiStepForwardDynamicsLSTM, CNNForwardDynamicsLSTM, CNNForwardDynamics, ForwardDynamicsLSTMParticles, PerceiverIOForwardDynamics
import torch

env = None
num_action_vals_two_pickers = 6
num_action_vals_one_picker = 3

def get_cost(args):
    init_state, action_trajs, env_class, env_kwargs, goal_obs = args
    global env
    if env is None:
        # Need to create the env inside the function such that the GPU buffer is associated with the child process and avoid any deadlock.
        # Use the global variable to access the child process-specific memory
        env = env_class(**env_kwargs)
        print('Child env created!')
    env.reset(config_id=init_state['config_id'])

    N = action_trajs.shape[0]
    costs = []
    for i in tqdm(range(N)):
        env.set_state(init_state)
        ret = 0
        for action in action_trajs[i, :]:
            next_obs, reward, _, _ = env.step(action)
            if env_kwargs['env'] == 'ThreeCubes':
                cost = reward
            else:
                # print('next_obs: ', next_obs[:-num_action_vals_one_picker])
                # print('goal_obs: ', goal_obs[:-num_action_vals_two_pickers])
                if len(goal_obs) == 18:
                    # reduced obs
                    cost = np.linalg.norm(next_obs[:-num_action_vals_one_picker] - goal_obs[:-num_action_vals_two_pickers])
                elif len(goal_obs) == 6400: 
                    # particles
                    cost = np.linalg.norm(env.env_pyflex.get_positions().reshape((-1, 4))[:, :3] - goal_obs)
            ret += cost
        costs.append(ret)
        # print('get_cost {}: {}'.format(i, ret))
    return costs

def get_obs(env, goal_obs, fwd_dyn_mode, particle_based_cnn_lstm_fwd_dyn_mode, particle_based_fwd_dyn_impl, extra_args):
    if fwd_dyn_mode == 'reduced_obs':
        obs = torch.from_numpy(env._get_obs()).float().cuda()[None, None, :-num_action_vals_one_picker]
        return obs
    elif fwd_dyn_mode == 'particles':
        particles = np.array(env.env_pyflex.get_positions()).reshape([-1, 4])[:, :3] # [num_particles, 3]

        if extra_args is not None and extra_args['downsample_idx'] is not None:
            particles = particles[extra_args['downsample_idx'], :]
            goal_obs = goal_obs[extra_args['downsample_idx'], :]

        particles = particles[None, None, :, :]
        particles = np.transpose(particles, (0, 1, 3, 2))
        if particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
            particles = np.reshape(particles, (particles.shape[0], particles.shape[1], particles.shape[2], int(np.sqrt(particles.shape[3])), int(np.sqrt(particles.shape[3]))))
            goal_obs = np.transpose(goal_obs, (1, 0))
            # convert to shape [xyz, clothdimx, clothdimy]
            goal_obs = np.reshape(goal_obs, (goal_obs.shape[0], int(np.sqrt(goal_obs.shape[1])), int(np.sqrt(goal_obs.shape[1]))))

        if particle_based_fwd_dyn_impl == 'cnn':
            # reshape to (1, 1, 3, 80, 80) from (1, 1, 3, 6400)
            particles = np.reshape(particles, (particles.shape[0], particles.shape[1], particles.shape[2], int(np.sqrt(particles.shape[3])), int(np.sqrt(particles.shape[3]))))
            # remove time_sequence dimension
            particles = particles[0, :, :, :, :]
        particles = torch.from_numpy(particles).float().cuda()
        return particles, goal_obs
    elif fwd_dyn_mode == 'state':
        obs = torch.from_numpy(env._get_obs()).float().cuda()[None, None, :]
        return obs

def get_cost_trained_fwd_dyn_helper(env, actions, particle_based_fwd_dyn_impl, lstm_based_methods, fwd_dyn_mode, forward_dyn_model, goal_obs, \
        num_action_vals_two_pickers, particle_based_cnn_lstm_fwd_dyn_mode, extra_args):
    ret = 0
    hs_forward = None

    # get initial env observation from Softgym simulator
    if fwd_dyn_mode == 'reduced_obs' or fwd_dyn_mode == 'state':
        obs = get_obs(env, goal_obs, fwd_dyn_mode, particle_based_cnn_lstm_fwd_dyn_mode, particle_based_fwd_dyn_impl, extra_args)
    elif fwd_dyn_mode == 'particles':
        particles, goal_obs = get_obs(env, goal_obs, fwd_dyn_mode, particle_based_cnn_lstm_fwd_dyn_mode, particle_based_fwd_dyn_impl, extra_args)
        # fold_group_a and fold_group_b are used to compute cloth fold reward
        # fold_group_a = extra_args['fold_group_a']
        # fold_group_b = extra_args['fold_group_b']
        # pos_group_b_init = particles.clone().cpu().detach().numpy()
        # if particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
        #     pos_group_b_init = pos_group_b_init[0, 0]
        #     pos_group_b_init = np.reshape(pos_group_b_init, (pos_group_b_init.shape[0], int(pos_group_b_init.shape[1] * pos_group_b_init.shape[2])))
        #     pos_group_b_init = pos_group_b_init[:, fold_group_b]
        #     pos_group_b_init = np.transpose(pos_group_b_init, (1, 0))
        # else:
        #     pos_group_b_init = pos_group_b_init[0, 0, :, fold_group_b]

    # iterate through a sequence of actions
    for action in actions:
        if particle_based_fwd_dyn_impl in lstm_based_methods or particle_based_fwd_dyn_impl == 'perceiverio':
            action = action[None, None, :]
        elif particle_based_fwd_dyn_impl == 'cnn':
            action = action[None, :]

        if fwd_dyn_mode == 'reduced_obs':
            st_act = torch.cat((obs, action), axis=2)
            obs, _ = forward_dyn_model(st_act, hs_forward, obs.shape)
            cost = np.linalg.norm(obs[0, 0, :].cpu().detach().numpy() - goal_obs[:-num_action_vals_two_pickers])
        elif fwd_dyn_mode == 'particles':
            if particle_based_fwd_dyn_impl in lstm_based_methods:
                pred_delta_pos_particles, hs_forward = forward_dyn_model(particles, action, hs_forward, particles.shape)
            elif particle_based_fwd_dyn_impl == 'cnn' or particle_based_fwd_dyn_impl == 'perceiverio':
                pred_delta_pos_particles = forward_dyn_model(particles, action, particles.shape)
            particles = pred_delta_pos_particles + particles
            if particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
                if particle_based_fwd_dyn_impl in lstm_based_methods:
                    particles_np = particles.clone().cpu().detach().numpy()[0, 0, :, :, :]
            else:
                particles_np = particles.clone().cpu().detach().numpy()[0, 0, :, :]
                particles_np = np.transpose(particles_np, (1, 0))

            if particle_based_fwd_dyn_impl == 'cnn':
                # remove time_sequence dimension
                particles_np = particles.clone().cpu().detach().numpy()[0, :, :, :]
                # reshape to (3, 6400) from (3, 80, 80)
                particles_np = np.reshape(particles_np, (particles_np.shape[0], int(particles_np.shape[1] * particles_np.shape[2])))
                # tranpose to (6400, 3) from (3, 6400)
                particles_np = np.transpose(particles_np, (1, 0))

            # particle distance cost
            cost = np.linalg.norm(particles_np - goal_obs)

            # cloth folding cost function
            # if particle_based_cnn_lstm_fwd_dyn_mode == '2dconv':
            #     particles_np = np.reshape(particles_np, (particles_np.shape[0], int(particles_np.shape[1] * particles_np.shape[2])))
            #     particles_np = np.transpose(particles_np, (1, 0))

            # pos_group_a = particles_np[fold_group_a]
            # pos_group_b = particles_np[fold_group_b]

            # top_left = particles_np[extra_args['corner1_idx']]
            # top_right = particles_np[extra_args['corner2_idx']]
            # bottom_left = particles_np[extra_args['corner3_idx']]
            # bottom_right = particles_np[extra_args['corner4_idx']]
            # corner_dist = (np.linalg.norm(top_left - top_right) + np.linalg.norm(bottom_left - bottom_right))

            # cost = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
            #     1.2 * np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1)) + corner_dist
        elif fwd_dyn_mode == 'state':
            # ThreeCubes environment
            obs, _ = forward_dyn_model(obs, action, hs_forward, obs.shape)
            # ignore timestep value in ThreeCubes' observations
            cost = np.linalg.norm(obs[0, 0, :3].cpu().detach().numpy() - goal_obs[:3])
        ret += cost
    return ret

def get_cost_trained_fwd_dyn(args):
    init_state, action_trajs, env_class, env_kwargs, goal_obs, forward_dyn_model, fwd_dyn_mode, particle_based_cnn_lstm_fwd_dyn_mode, particle_based_fwd_dyn_impl, extra_args = args
    global env
    if env is None:
        # Need to create the env inside the function such that the GPU buffer is associated with the child process and avoid any deadlock.
        # Use the global variable to access the child process-specific memory
        env = env_class(**env_kwargs)
        print('Child env created!')
    env.reset(config_id=init_state['config_id'])
    action_trajs = torch.from_numpy(action_trajs).float().cuda()

    N = action_trajs.shape[0]
    costs = []
    lstm_based_methods = ['cnn_lstm' , 'lstm_particles']
    num_fdy_model_interactions = 0
    for num_samples in range(N):
        env.set_state(init_state) 
        ret = get_cost_trained_fwd_dyn_helper(env, action_trajs[num_samples, :], particle_based_fwd_dyn_impl, lstm_based_methods, fwd_dyn_mode, forward_dyn_model, goal_obs, \
            num_action_vals_two_pickers, particle_based_cnn_lstm_fwd_dyn_mode, extra_args)
        costs.append(ret)
        num_fdy_model_interactions += len(action_trajs[num_samples, :])
    print(f'min cost: {min(costs)}; max cost: {max(costs)}')
    return costs, num_fdy_model_interactions

class ParallelRolloutWorker(object):
    """ Rollout a set of trajectory in parallel. """

    def __init__(self, env_class, env_kwargs, plan_horizon, action_dim, kwargs, num_worker=8):
        # self.num_worker = 1 # debug
        self.num_worker = num_worker
        self.plan_horizon, self.action_dim = plan_horizon, action_dim
        self.env_class, self.env_kwargs = env_class, env_kwargs
        self.enable_trained_fwd_dyn = kwargs.get('enable_trained_fwd_dyn')
        self.particle_based_cnn_lstm_fwd_dyn_mode = kwargs.get('particle_based_cnn_lstm_fwd_dyn_mode')
        self.particle_based_fwd_dyn_impl = kwargs.get('particle_based_fwd_dyn_impl')
        self.enable_downsampling = kwargs.get('enable_downsampling')
        self.env_name = env_kwargs['env']
        if self.enable_trained_fwd_dyn:
            pretrained_fwd_dyn_ckpt = kwargs.get('pretrained_fwd_dyn_ckpt')
            assert pretrained_fwd_dyn_ckpt is not None
            self.env = env_class(**env_kwargs)
            self.hidden_size = 32
            self.fwd_dyn_mode = kwargs.get('fwd_dyn_mode')
            if self.fwd_dyn_mode == 'reduced_obs':
                self.forward_dyn_model = MultiStepForwardDynamicsLSTM(self.env.observation_space.shape[0] - num_action_vals_one_picker, self.env.action_space.shape[0], hidden_size=self.hidden_size)
                self.extra_args = None
            elif self.fwd_dyn_mode == 'particles':
                cloth_dim_xandy = self.env._sample_cloth_size()[0]
                self.downsample_idx = None
                if self.enable_downsampling:
                    down_sample_scale = 4
                    new_idx = np.arange(cloth_dim_xandy * cloth_dim_xandy).reshape((cloth_dim_xandy, cloth_dim_xandy))
                    new_idx = new_idx[::down_sample_scale, ::down_sample_scale]
                    new_cloth_ydim, new_cloth_xdim = new_idx.shape
                    num_particles = new_cloth_ydim * new_cloth_xdim
                    self.downsample_idx = new_idx.flatten()
                    self.extra_args = {
                        'downsample_idx': self.downsample_idx,
                    }
                else:
                    if 'Cloth' in env_kwargs['env']:
                        # fold_group_a and fold_group_b are used to compute cloth fold reward
                        num_particles = int(cloth_dim_xandy * cloth_dim_xandy)
                        particle_grid_idx = np.array(list(range(num_particles))).reshape(cloth_dim_xandy, cloth_dim_xandy)  # Reversed index here
                        x_split = cloth_dim_xandy // 2
                        fold_group_a = particle_grid_idx[:, :x_split].flatten()
                        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()
                        self.extra_args = {
                            'fold_group_a': fold_group_a,
                            'fold_group_b': fold_group_b,
                            'corner1_idx': particle_grid_idx[0, 0],
                            'corner2_idx': particle_grid_idx[0, -1],
                            'corner3_idx': particle_grid_idx[-1, 0],
                            'corner4_idx': particle_grid_idx[-1, -1],
                            'downsample_idx': self.downsample_idx,
                        }
                    elif 'Rope' in env_kwargs['env']:
                        num_particles = 41
                        self.extra_args = None
                    else:
                        raise NotImplementedError
                if self.particle_based_fwd_dyn_impl == 'cnn_lstm':
                    self.forward_dyn_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0] - num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size)
                elif self.particle_based_fwd_dyn_impl == 'cnn':
                    self.forward_dyn_model = CNNForwardDynamics(self.env.observation_space.shape[0] - num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, hidden_size=self.hidden_size)
                elif self.particle_based_fwd_dyn_impl == 'lstm_particles':
                    self.forward_dyn_model = ForwardDynamicsLSTMParticles(self.env.observation_space.shape[0] - num_action_vals_one_picker, self.env.action_space.shape[0], num_particles, hidden_size=self.hidden_size)
                elif self.particle_based_fwd_dyn_impl == 'perceiverio':
                    self.forward_dyn_model = PerceiverIOForwardDynamics(self.env.action_space.shape[0], num_particles)
            elif self.fwd_dyn_mode == 'state':
                self.forward_dyn_model = CNNForwardDynamicsLSTM(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.env.observation_space.shape[0], self.particle_based_cnn_lstm_fwd_dyn_mode, hidden_size=self.hidden_size, env_kwargs=env_kwargs)
                self.extra_args = None

            checkpoint = torch.load(pretrained_fwd_dyn_ckpt, map_location='cpu')
            self.forward_dyn_model.load_state_dict(checkpoint['forward_state_dict'])
            for param in self.forward_dyn_model.parameters():
                param.requires_grad = False
            self.forward_dyn_model.cuda()
            self.forward_dyn_model.eval()
            print('Loaded pre-trained weights to the forward dynamics model!')

        # self.pool = Pool(processes=num_worker) # this is actually slower than not using multi-processing

    def cost_function(self, init_state, action_trajs, goal_obs, curr_env_timestep):
        action_trajs = action_trajs.reshape([-1, self.plan_horizon, self.action_dim])
        # splitted_action_trajs = np.array_split(action_trajs, self.num_worker)
        # ret = self.pool.map(get_cost, [(init_state, splitted_action_trajs[i], self.env_class, self.env_kwargs, goal_obs) for i in range(self.num_worker)])

        if self.env_name == 'ClothFold':
            action_trajs[:, :, 0] = np.clip(action_trajs[:, :, 0], -0.5, 0.5) # sim_min, sim_max (normalized between -1 and 1)
            action_trajs[:, :, 1] = -0.9714286 # optimal pick height (normalized)
            action_trajs[:, :, 2] = np.clip(action_trajs[:, :, 2], -0.5, 0.5) # sim_min, sim_max (normalized between -1 and 1)
            action_trajs[:, :, 3] = np.clip(action_trajs[:, :, 3], -0.5, 0.5) # sim_min, sim_max (normalized between -1 and 1)
            action_trajs[:, :, 4] = -0.71428573 # optimal place height (normalized)
            action_trajs[:, :, 5] = np.clip(action_trajs[:, :, 5], -0.5, 0.5) # sim_min, sim_max (normalized between -1 and 1)
        elif self.env_name == 'RopeFlatten':
            if curr_env_timestep == 0:
                # since rope always start in this range, we want to make the search space smaller.
             action_trajs[:, :, 0] = np.clip(action_trajs[:, :, 0], -0.2, 0.2)
             action_trajs[:, :, 2] = np.clip(action_trajs[:, :, 2], -0.2, 0.2)
            action_trajs[:, :, 1] = -0.9714286 # optimal pick height (normalized)
            action_trajs[:, :, 4] = -0.71428573 # optimal place height (normalized)
        elif self.env_name == 'DryCloth':
            # best pick and place bounds:
            # try 1 min_x: -0.9 max_x: -0.7
            # try 2 min_x: -1.0, max_x: -0.7
            # try 1 min_z: -0.8, max_z: 1
            # try 2 min_z: -0.6, max_z: 0.6
            if curr_env_timestep == 0:
                action_trajs[:, :, 1] = -0.9714286 # optimal pick height for picking up cloth on table (normalized)
            else:
                action_trajs[:, :, 0] = np.clip(action_trajs[:, :, 0], -1.0, -0.7)  # normalized
                action_trajs[:, :, 1] = 0 # pick height for picking up cloth on rack
                action_trajs[:, :, 2] = np.clip(action_trajs[:, :, 2], -0.6, 0.6)  # normalized

            action_trajs[:, :, 3] = np.clip(action_trajs[:, :, 3], -1.0, -0.7)    # normalized
            action_trajs[:, :, 4] = 0.7                                           # normalized
            action_trajs[:, :, 5] = np.clip(action_trajs[:, :, 5], -0.6, 0.6)       # normalized

            # old values
            # action_trajs[:, :, 3] = np.clip(action_trajs[:, :, 3], -1.4, -0.8)  # normalized
            # action_trajs[:, :, 4] = 0.7                                         # normalized
            # action_trajs[:, :, 5] = np.clip(action_trajs[:, :, 5], -1, 1)       # normalized
            # self.env.normalize_action(np.array([0, 0, 0, -0.7, 0.595, -0.5])) # min_place
            # self.env.normalize_action(np.array([0, 0, 0, -0.4, 0.595, 0.5]))  # max_place
        elif self.env_name == 'ThreeCubes':
            action_trajs[:, :, :] = np.clip(action_trajs[:, :, :], -0.1, 1.35) # sim_min, sim_max (normalized between -1 and 1)
        else:
            raise NotImplementedError

        num_fdy_model_interactions = 0
        if self.enable_trained_fwd_dyn:
            ret, num_fdy_model_interactions = get_cost_trained_fwd_dyn((init_state, action_trajs, self.env_class, self.env_kwargs, goal_obs, self.forward_dyn_model, self.fwd_dyn_mode, self.particle_based_cnn_lstm_fwd_dyn_mode, self.particle_based_fwd_dyn_impl, self.extra_args)) # no worker used
        else:
            ret = get_cost((init_state, action_trajs, self.env_class, self.env_kwargs, goal_obs)) # no worker used
        # flat_costs = [item for sublist in ret for item in sublist]  # ret is indexed first by worker_num then traj_num
        flat_costs = [item for item in ret]
        return flat_costs, num_fdy_model_interactions
