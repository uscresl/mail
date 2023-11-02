import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt


def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration', 'DryCloth']
    parser.add_argument('--env_name', type=str, default='DryCloth')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--save_video_pickplace', action='store_true', default=False, help='Whether to save pick and place recorded videos')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--expert', action='store_true', default=False, help='Whether to use the expert policy')
    parser.add_argument('--num_picker', type=int, default=None, help='Overwrite num_picker in the environment')
    parser.add_argument('--action_mode', type=str, default=None, help='Overwrite action_mode in the environment')
    parser.add_argument('--action_repeat', type=int, default=None, help='Overwrite action_repeat in the environment')
    parser.add_argument('--env_horizon', type=int, default=None, help='Set a non-default number of steps for the episode')
    parser.add_argument('--cam_view', default='side', choices=['top_down', 'side'])
    parser.add_argument('--save_cached_states', action='store_true', default=False, help='Whether to save cached_init_states')
    parser.add_argument('--use_cached_states', action='store_true', default=False, help='Whether to use cached_init_states')
    parser.add_argument('--enable_animations', action='store_true', default=False, help="Whether to enable animations during evaluations") # only for ThreeCubes environment


    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = args.use_cached_states
    env_kwargs['save_cached_states'] = args.save_cached_states
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['render_mode'] = 'particle'
    env_kwargs['cam_view'] = args.cam_view

    if args.num_picker:
        env_kwargs['num_picker'] = args.num_picker
    if args.action_mode:
        env_kwargs['action_mode'] = args.action_mode
    if args.action_repeat:
        env_kwargs['action_repeat'] = args.action_repeat
    if args.env_horizon:
        env_kwargs['horizon'] = args.env_horizon
    if args.env_name == 'ThreeCubes' and args.enable_animations:
        env_kwargs['enable_animations'] = args.enable_animations

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()
    frames = [env.get_image(args.img_size, args.img_size)]
    if args.save_video_pickplace:
        env.start_record()

    # n_particles = pyflex.get_n_particles()
    # cloth_dimx = cloth_dimy = env.current_config['ClothSize'][1]
    # midair_pick = env.action_space.sample()*0 # effectively doing nothing
    # pick_center_place_center = np.array([0., -1, 0., -0.75, 0.35, 0.])
    # pick_topleft_place_center = np.array([-0.3687, -0.9857, -0.3687, -1, 0.65, -0.5])


    for i in range(env.horizon):
        if args.expert:
            action = env.compute_expert_action()
        else:
            action = env.action_space.sample()
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        # if args.num_picker == 1:
            # pos = pyflex.get_positions().reshape(-1,4)
            # particle_grid_idx = np.array(list(range(n_particles))).reshape(cloth_dimx, cloth_dimy)
            # corner_idxs = [particle_grid_idx[0,0], particle_grid_idx[0,-1], particle_grid_idx[-1,0], particle_grid_idx[-1,-1]]
            # print(f'corner_idxs: {corner_idxs}')
            # print(f'corner picks {[env.normalize_action(env, pos[idx][0:3]) for idx in corner_idxs]}')
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        print(f'normalized performance: {info["normalized_performance"]}')
        # print(f'step {i}, normalized_performance {info["normalized_performance"]}')
        frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth:
            show_depth()

    if args.save_video_pickplace:
        env.end_record(video_path=f'./data/{args.env_name}.gif')
    elif args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()