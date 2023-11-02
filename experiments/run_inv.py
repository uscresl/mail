from softgym.registered_env import env_arg_dict
from sb3.utils import str2bool, set_seed_everywhere, update_env_kwargs, make_dir, NumpyEncoder
import torch
import argparse
import json
from datetime import datetime
# from inv.model_v2 import InverseDynamicsModel
from inv.model_v3 import DynamicsModel

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
    'ThreeCubes': 50.0,
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
    'ThreeCubes': None,
}

def main():
    parser = argparse.ArgumentParser()

    ############## Experiment ##############
    # evaluation arguments
    parser.add_argument('--is_eval',        default=False, type=str2bool, help="evaluation or training mode")
    parser.add_argument('--checkpoint',     default=None, type=str, help="checkpoint file for evaluation")
    parser.add_argument('--num_eval_eps',   default=10, type=int, help="number of episodes to run during evaluation")
    parser.add_argument('--eval_videos',    default=False, type=str2bool, help="whether or not to save evaluation video per episode")
    parser.add_argument('--eval_gif_size',  default=256, type=int, help="evaluation GIF width and height size")
    parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

    # logging arguments
    parser.add_argument('--wandb',          action='store_true', help="use wandb instead of tensorboard for logging")

    # validation arguments
    parser.add_argument('--eval_interval', default=1, type=int, help="Evaluation interval (in terms of epochs) during training")

    # task arguments
    parser.add_argument('--env_name',       default='ClothFold')

    # model arguments
    parser.add_argument('--batch_size',     default=256, type=int, help="training batch_size")
    parser.add_argument('--name',           default=None, type=str, help='[optional] set experiment name. Useful to resume experiments.')
    parser.add_argument('--seed',           default=1234, type=int, help="seed number")
    parser.add_argument('--epoch',          default=1_000, type=int, help="number of training epochs")
    parser.add_argument('--two_arms_expert_data', default=None, type=str, help='Two arms demonstration data')
    parser.add_argument('--random_actions_data', action='append', help='List of random actions data')
    parser.add_argument('--num_actions',    default=6, type=int, help="predefined number of output actions for inverse dynamics model")
    parser.add_argument('--enable_fine_tuning', default=False, type=str2bool, help="whether or not to enable fine tuning (training on expert dataset)")
    parser.add_argument('--learning_rate', default=1e-3, type=float, help="learning rate for inverse and forward dynamics models")
    parser.add_argument('--resume_training', default=False, type=str2bool, help="continue training ")
    parser.add_argument('--cem_trajs_folder', default=None, type=str, help='Path to CEM trajectories folder')
    parser.add_argument('--train_mode', default='fwd', choices=['fwd', 'inv', 'inv_fwd', 'fine_tune'], help='fwd: pre-trains forward dynamics model; inv: trains an inverse dynamics model with frozen forward dynamics model; inv_fwd: jointly trains an inverse dynamics model and a forward dynamics model')
    parser.add_argument('--pretrain_fwd_model_ckpt', default=None, type=str, help="pre-trained forward dynamics model checkpoint file")
    parser.add_argument('--pretrain_inv_model_ckpt', default=None, type=str, help="pre-trained inverse dynamics model checkpoint file")
    parser.add_argument('--enable_inv_dyn_mse_loss', default=True, type=str2bool, help="whether to use inverse dynamics model's mse loss")
    parser.add_argument('--enable_particle_based_fwd_dyn', default=False, type=str2bool, help="whether to use particle-based forward dynamics model")
    parser.add_argument('--enable_downsampling', default=False, type=str2bool, help="Whether to downsample cloth")
    parser.add_argument('--particle_based_fwd_dyn_impl', default='cnn_lstm', choices=['cnn_lstm', 'cnn', 'lstm_reduced_obs', 'lstm_particles', 'perceiverio'], help='Choose a particle-based forward dynamics model implementation')
    parser.add_argument('--particle_based_cnn_lstm_fwd_dyn_mode', default='1dconv', choices=['1dconv', '2dconv'], help='Choose a particle-based forward dynamics model implementation')
    parser.add_argument('--should_freeze_fwd_model', default=False, type=str2bool, help="whether to freeze the forward dynamics model")
    parser.add_argument('--extract_num_time_step_data', default=None, type=int, help="Extract the number of time steps data from the dataset")
    parser.add_argument('--visualize_fwd_model', default=False, type=str2bool, help="whether to visualize the forward dynamics model")


    ############## Override environment arguments ##############
    parser.add_argument('--env_kwargs_render', default=True, type=str2bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']. Only AWAC supports 'cam_rgb_key_point' and 'depth_key_point'.
    parser.add_argument('--env_kwargs_num_variations', default=100, type=int)
    parser.add_argument('--env_kwargs_env_image_size', default=32, type=int, help="observation image size")
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int, help='Number of pickers/end-effectors')
    parser.add_argument('--action_mode', type=str, default='pickerpickandplace', help='Overwrite action_mode in the environment')
    parser.add_argument('--action_repeat', type=int, default=1, help='Overwrite action_repeat in the environment')
    parser.add_argument('--horizon', type=int, default=6, help='Overwrite action_repeat in the environment')

    args = parser.parse_args()

    # Set env_specific parameters
    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]
    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    not_imaged_based = args.env_kwargs['observation_mode'] not in ['cam_rgb', 'cam_rgb_key_point', 'depth_key_point']
    symbolic = not_imaged_based
    args.encoder_type = 'identity' if symbolic else 'pixel'
    args.max_steps = 200
    env_kwargs = {
        'env': args.env_name,
        'symbolic': symbolic,
        'seed': args.seed,
        'max_episode_length': args.max_steps,
        'action_repeat': 1,
        'bit_depth': 8,
        'image_dim': None if not_imaged_based else args.env_kwargs['env_image_size'],
        'env_kwargs': args.env_kwargs,
        'normalize_observation': False,
        'scale_reward': args.scale_reward,
        'clip_obs': args.clip_obs,
        'obs_process': None,
    }
    env_kwargs['env_kwargs']['action_mode'] = args.action_mode
    env_kwargs['env_kwargs']['action_repeat'] = args.action_repeat
    env_kwargs['env_kwargs']['horizon'] = args.horizon


    # assertions
    assert args.horizon == args.num_actions
    if args.enable_fine_tuning or args.resume_training:
        assert args.checkpoint # ensure pretrained checkpoint is provided

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"Device set to {device}")

    set_seed_everywhere(args.seed)

    if args.visualize_fwd_model:
        assert args.checkpoint is not None
        agent = DynamicsModel(args.__dict__, env_kwargs)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        agent.forward_dynamics_model_visualization(args, checkpoint)
    elif args.is_eval:
        assert args.two_arms_expert_data is not None
        agent = DynamicsModel(args.__dict__, env_kwargs)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        agent.evaluate(args, checkpoint)
    else:
        now = datetime.now().strftime("%m.%d.%H.%M")
        args.folder_name = f'{env_name}_InvDynModel_{now}' if not args.name else args.name
        args.tb_dir = f"./data/followup/{args.folder_name}"
        args.ckpt_saved_folder = f'{args.tb_dir }/checkpoints/'
        make_dir(f'{args.ckpt_saved_folder}')
        with open(f'{args.tb_dir}/config.json', 'w') as outfile:
            json.dump(args.__dict__, outfile, indent=2, cls=NumpyEncoder)

        agent = DynamicsModel(args.__dict__, env_kwargs)
        agent.run(args)

print(f"Done! Train/eval script finished.")

if __name__ == '__main__':
    main()
