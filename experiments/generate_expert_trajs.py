import argparse
import os.path as osp
import pickle

import numpy as np
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from sb3.utils import str2bool
from tqdm import tqdm
from sb3.utils import make_dir, set_seed_everywhere

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
    parser.add_argument('--remove_actions_from_obs', type=str2bool, default=False, help='Whether to remove action values from observations (default observations contain action values)')
    parser.add_argument('--save_gif', type=str2bool, default=False, help='Whether to save results as gif')
    parser.add_argument('--save_observation_img', type=str2bool, default=False, help='Whether to save observation image (for image-based methods)')
    parser.add_argument('--save_particles', type=str2bool, default=False, help='Whether to save particle positions')
    parser.add_argument('--image_mode', default='rgb', choices=['rgb', 'depth'], help='Whether the observation image is depth or rgb')
    parser.add_argument('--save_states_in_folder', type=str2bool, default=False, help='Whether to save states in a separate folder')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
    parser.add_argument('--cam_view', default='side', choices=['top_down', 'side'])
    parser.add_argument('--ir_reward_weight', default=0, type=float, help='weight for imitation reward')
    parser.add_argument('--out_filename', default=None, type=str)
    parser.add_argument('--seed', default=1234, type=int, help="seed number")

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['observation_mode'] = args.observation_mode # Always state for experts.
    env_kwargs['ir_reward_weight'] = args.ir_reward_weight
    env_kwargs['cam_view'] = args.cam_view
    # if args.env_kwargs_observation_mode != 'point_cloud':
    #     raise Warning('Obs mode is not point_cloud (i.e. not getting full state info')

    set_seed_everywhere(args.seed)

    configs = []
    state_trajs = []
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
    episodes_saved = 0

    filename = args.out_filename if args.out_filename else f'{args.env_name}_numvariations{args.num_variations}_eps{args.num_eps}_trajs.pkl'
    filepath = osp.join(args.save_dir, filename)

    if args.save_states_in_folder:
        states_folder = filepath.replace('.pkl', '_states')
        make_dir(states_folder)
        states_id = 0

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
    #assert isinstance(env._wrapped_env, RopeFlattenEnv), "Expert policy is only available for RopeFlattenEnv"
    #assert isinstance(env._wrapped_env, ClothFoldEnv), "Expert policy is only available for ClothFoldEnv"

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
        states = []
        actions = []
        observations = []
        next_observations = []
        particles = []
        next_particles = []
        rewards = []
        dones = []
        ob_img_traj = []
        next_ob_img_traj = []
        obs = env._get_obs()
        if args.save_gif:
            if args.action_mode == 'pickerpickandplace':
                env.start_record()
            frames = [env.get_image(args.img_size, args.img_size)]

        for i in range(env.horizon):
            action = env.compute_expert_action()
            if args.save_states_in_folder:
                states_file_path = osp.join(states_folder, str(states_id) + '.npy')
                cur_state = env.get_state()
                np.save(states_file_path, cur_state)
                states.append(states_file_path)
                states_id += 1
            else:
                states.append(env.get_state())
            actions.append(action)
            if args.remove_actions_from_obs:
                obs = obs[:-num_action_vals]
            observations.append(obs)
            if args.save_particles:
                particles.append(env.env_pyflex.get_positions().reshape(-1, 4)[:, :3])
            if args.save_observation_img:
                if args.image_mode == 'rgb':
                    ob_img_traj.append(env.get_image(args.env_img_size, args.env_img_size))
                elif args.image_mode == 'depth':
                    ob_img_traj.append(env.get_depth_image(args.env_img_size, args.env_img_size))
            if args.save_gif:
                # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
                # intermediate frames. Only use this option for visualization as it increases computation.
                obs, rew, done, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
                frames.extend(info['flex_env_recorded_frames'])
            else:
                obs, rew, done, info = env.step(action)

            if args.save_observation_img:
                if args.image_mode == 'rgb':
                    cur_img = env.get_image(args.env_img_size, args.env_img_size)
                elif args.image_mode == 'depth':
                    cur_img = env.get_depth_image(args.env_img_size, args.env_img_size)
                next_ob_img_traj.append(cur_img)
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


        if args.save_gif:
            #save_name = osp.join(args.save_dir, f'{args.env_name}_{eps}.gif')
            save_name = osp.join(args.save_dir, f'{args.env_name}_{episodes_saved}.gif')
            save_numpy_as_gif(np.array(frames), save_name)
            print('Video generated and save to {}'.format(save_name))

            if args.action_mode == 'pickerpickandplace':
                env.end_record(video_path=osp.join(args.save_dir, f'{args.env_name}_pickandplace_{episodes_saved}.gif'))
            episodes_saved += 1

        if args.env_name == 'ClothFlatten' and ep_normalized_perf[-1] < 0.2:
            # only save trajs if info_normalized_performance_final is >= 0.2
            continue

        total_normalized_performance.append(ep_normalized_perf[-1]) #only store performance at the end of the episode
        configs.append(env.get_current_config().copy())
        state_trajs.append(states)
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
        eps += 1
        pbar.update(1)
        if eps >= args.num_eps:
            is_done_generating = True

    if args.save_dir:
        traj_data = dict(configs=configs,
        state_trajs=np.array(state_trajs),
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