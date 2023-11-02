import argparse
import os.path as osp
import pickle

import numpy as np
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict
from softgym.utils.normalized_env import normalize
from sb3.utils import str2bool, set_seed_everywhere
from tqdm import tqdm
import drq.utils as utils
import os

def main():
    parser = argparse.ArgumentParser()
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='RopeFlatten')
    parser.add_argument('--env_img_size', type=int, default=128, help='Environment (observation) image size (only used if save_observation_img is True)')
    parser.add_argument('--observation_mode', type=str, default='key_point', help='Observation mode')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1000, help='Number of environment variations')
    parser.add_argument('--num_eps', type=int, default=100000, help='Number of epsidoes to be generated')
    parser.add_argument('--env_horizon', type=int, default=None, help='Set a non-default number of steps for the episode')
    parser.add_argument('--action_mode', type=str, default=None, help='Overwrite action_mode in the environment')
    parser.add_argument('--action_repeat', type=int, default=None, help='Overwrite action_repeat in the environment')
    parser.add_argument('--num_picker', default=None, type=int, help='Overwrite num_picker in the environment')
    parser.add_argument('--save_dir', type=str, default='./data/', help='Path to the saved video/demonstrations data')
    parser.add_argument('--save_particles', type=str2bool, default=False, help='Whether to save particle positions')
    parser.add_argument('--eval_videos', type=str2bool, default=False, help='Whether to save evaluation video')
    parser.add_argument('--eval_video_path', type=str, default='./data/debug_videos')
    parser.add_argument('--save_every_x_timesteps', type=int, default=None, help='Save transition every x timesteps')
    parser.add_argument('--remove_actions_from_obs', type=str2bool, default=False, help='Whether to remove action values from observations (default observations contain action values)')
    parser.add_argument('--cam_view', default='side', choices=['top_down', 'side'])
    parser.add_argument('--truly_random', type=str2bool, default=False, help='Whether to sample actions from the action space (truly random)')
    parser.add_argument('--two_arms_expert_data', default=None, type=str, help='File path to two-arms-teacher dataset')
    parser.add_argument('--save_observation_img', type=str2bool, default=False, help='Whether to save observation image (for image-based methods)')
    parser.add_argument('--seed', default=1234, type=int, help="seed number")
    parser.add_argument('--out_filename', default=None, type=str)

    args = parser.parse_args()

    # assertions
    if args.save_every_x_timesteps:
        assert args.env_horizon is not None
        assert args.save_every_x_timesteps < args.env_horizon
        if args.env_horizon % args.save_every_x_timesteps != 0:
            print('save_every_x_timesteps must be dively divisble by env_horizon')
            exit(-1)

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['observation_mode'] = args.observation_mode # Always state for experts.
    env_kwargs['cam_view'] = args.cam_view

    set_seed_everywhere(args.seed)

    configs = []
    action_trajs = []
    ob_trajs = []
    ob_next_trajs = []
    particles_trajs = []
    particles_next_trajs = []
    reward_trajs = []
    done_trajs = []
    ob_img_trajs = []
    ob_img_next_trajs = []
    ep_rews = []
    total_normalized_performance = []

    filename = args.out_filename if args.out_filename else f'{args.env_name}_numvariations{args.num_variations}_eps{args.num_eps}_trajs.pkl'
    filepath = osp.join(args.save_dir, filename)

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    if args.env_horizon:
        env_kwargs['horizon'] = args.env_horizon
    if args.action_mode:
        env_kwargs['action_mode'] = args.action_mode
    if args.action_repeat:
        env_kwargs['action_repeat'] = args.action_repeat
    if args.num_picker:
        env_kwargs['num_picker'] = args.num_picker

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))

    if hasattr(env, 'action_tool'):
        num_picker = env.action_tool.num_picker
    else:
        num_picker = args.num_picker
    num_action_vals = num_picker * 3 # each picker has x, y, z values
    pbar = tqdm(total=args.num_eps)
    eps = 0
    is_done_generating = False
    while not is_done_generating:
        env.reset()
        ep_rew = 0.
        ep_normalized_perf = []
        actions = []
        observations = []
        next_observations = []
        particles = []
        next_particles = []
        ob_img_traj = []
        next_ob_img_traj = []
        rewards = []
        dones = []
        obs = env._get_obs()

        if args.eval_videos:
            transition_num = 1
            env.start_record()

        if args.truly_random:
            with open(args.two_arms_expert_data, 'rb') as f:
                two_arms_data = pickle.load(f)
                two_arms_data_configs = two_arms_data['configs']
                two_arms_data_state_trajs = two_arms_data['state_trajs']
            
            for target_ep_index in range(len(two_arms_data_configs)):
                env.reset()
                env.set_scene(two_arms_data_configs[target_ep_index], two_arms_data_state_trajs[target_ep_index][0])
                obs = env._reset()

                ep_normalized_perf = []
                done = False

                while not done:
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    ep_normalized_perf.append(info['normalized_performance'])
                total_normalized_performance.append(ep_normalized_perf[-1])
                print(f'Ep {target_ep_index}  Episode normalized performance final: {ep_normalized_perf[-1]}')
                pbar.update(1)
            is_done_generating = True       
        else:
            for i in range(1, env.horizon+1):
                action = env.compute_random_actions()
                actions.append(action)
                if args.remove_actions_from_obs:
                    obs = obs[:-num_action_vals]
                observations.append(obs)
                if args.save_particles:
                    particles.append(env.env_pyflex.get_positions().reshape(-1, 4)[:, :3])
                if args.save_observation_img:
                    ob_img_traj.append(env.get_image(args.env_img_size, args.env_img_size))
                obs, rew, done, info = env.step(action)

                if args.save_observation_img:
                    next_ob_img_traj.append(env.get_image(args.env_img_size, args.env_img_size))

                if args.remove_actions_from_obs:
                    next_obs = obs[:-num_action_vals]
                else:
                    next_obs = obs
                next_observations.append(next_obs)
                if args.save_particles:
                    next_particles.append(env.env_pyflex.get_positions().reshape(-1, 4)[:, :3])
                rewards.append(rew)
                dones.append(done)
                ep_rew += rew
                ep_normalized_perf.append(info['normalized_performance'])

                if args.save_every_x_timesteps and i % args.save_every_x_timesteps == 0:
                    # store this partial trajectory as our transition (for training LSTM model)
                    total_normalized_performance.append(ep_normalized_perf[-1])
                    configs.append(env.get_current_config().copy())
                    action_trajs.append(actions)
                    ob_trajs.append(observations)
                    ob_next_trajs.append(next_observations)
                    particles_trajs.append(particles)
                    particles_next_trajs.append(next_particles)
                    reward_trajs.append(rewards)
                    done_trajs.append(dones)
                    ep_rews.append(ep_rew)
                    if args.save_observation_img:
                        ob_img_trajs.append(np.array(ob_img_traj.copy()))
                        ob_img_next_trajs.append(np.array(next_ob_img_traj.copy()))

                    # clear variables so that we can obtain states in the mid-trajectory
                    ep_rew = 0.
                    ep_normalized_perf = []
                    actions = []
                    observations = []
                    next_observations = []
                    particles = []
                    next_particles = []
                    ob_img_traj = []
                    next_ob_img_traj = []
                    rewards = []
                    dones = []

                    if args.eval_videos:
                        env.end_record(video_path=os.path.join(args.eval_video_path, f'transition_{transition_num}.gif'))
                        transition_num += 1
                        env.start_record()
        eps += 1
        pbar.update(1)
        if eps >= args.num_eps:
            is_done_generating = True

    if args.save_dir:
        traj_data = dict(configs=configs,
        action_trajs=np.array(action_trajs),
        ob_trajs=np.array(ob_trajs),
        ob_next_trajs=np.array(ob_next_trajs),
        particles_trajs=np.array(particles_trajs),
        particles_next_trajs=np.array(particles_next_trajs),
        ob_img_trajs=np.array(ob_img_trajs),
        ob_img_next_trajs=np.array(ob_img_next_trajs),
        reward_trajs=np.array(reward_trajs),
        done_trajs=np.array(done_trajs),
        ep_rews=np.array(ep_rews),
        total_normalized_performance=np.array(total_normalized_performance),
        env_kwargs=env_kwargs)
        with open(filepath, 'wb') as fh:
            pickle.dump(traj_data, fh, protocol=4)
        print(f'{args.num_variations} environment variations and {args.num_eps} episodes generated and saved to {filepath}')

    normalized_performance_final = np.mean(total_normalized_performance)
    print(f'info_normalized_performance_final: {normalized_performance_final}')

if __name__ == '__main__':
    main()