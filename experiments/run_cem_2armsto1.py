from cem_2armsto1.cem import CEMPolicy
from experiments.planet.train import update_env_kwargs
from cem_2armsto1.visualize_cem import cem_make_gif
from planet.utils import transform_info
from envs.env import Env
from chester import logger
import torch
import pickle
import os
import os.path as osp
import copy
import multiprocessing as mp
import json
import numpy as np
from softgym.registered_env import env_arg_dict
from datetime import datetime
import wandb
from sb3.utils import str2bool
import random
from curl import utils

def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    return args


cem_plan_horizon = {
    'ClothFold': 2,
    'DryCloth': 2,
    'RopeFlatten': 2,
    'ThreeCubes': 2,
}


def run_task(vv, log_dir, exp_name):
    mp.set_start_method('spawn')
    env_name = vv['env_name']
    vv['algorithm'] = 'CEM'
    vv['env_kwargs'] = env_arg_dict[env_name]  # Default env parameters
    vv['plan_horizon'] = cem_plan_horizon[env_name] if vv['cem_plan_horizon'] is None else vv['cem_plan_horizon']# Planning horizon
    print('plan_horizon: ', vv['plan_horizon'])
    print('max_iters: ', vv['max_iters'])
    print('timestep_per_decision: ', vv['timestep_per_decision'])

    vv['population_size'] = vv['timestep_per_decision'] // vv['max_iters']
    if vv['use_mpc']:
        vv['population_size'] = vv['population_size'] // vv['plan_horizon']
    vv['num_elites'] = vv['population_size'] // 10
    vv = update_env_kwargs(vv)

    # Configure logger
    if vv['is_eval'] == False:
        logger.configure(dir=log_dir, exp_name=exp_name)
        logdir = logger.get_dir()
        assert logdir is not None
        os.makedirs(logdir, exist_ok=True)

        # Dump parameters
        with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
            json.dump(vv, f, indent=2, sort_keys=True)

    # Configure torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(vv['seed'])

    utils.set_seed_everywhere(vv['seed'])

    env_symbolic = vv['env_kwargs']['observation_mode'] != 'cam_rgb'
    vv['env_kwargs']['cam_view'] = vv['cam_view']

    if vv['env_name'] == 'ThreeCubes' and vv['enable_animations']:
        vv['env_kwargs']['enable_animations'] = True

    env_class = Env
    env_kwargs = {'env': vv['env_name'],
                  'symbolic': env_symbolic,
                  'seed': vv['seed'],
                  'max_episode_length': 200,
                  'action_repeat': 1,  # Action repeat for env wrapper is 1 as it is already inside the env
                  'bit_depth': 8,
                  'image_dim': None,
                  'env_kwargs': vv['env_kwargs']}
    env = env_class(**env_kwargs)

    env_kwargs_render = copy.deepcopy(env_kwargs)
    env_kwargs_render['env_kwargs']['render'] = True
    env_render = env_class(**env_kwargs_render)

    policy = CEMPolicy(env, env_class, env_kwargs, vv['use_mpc'], plan_horizon=vv['plan_horizon'], max_iters=vv['max_iters'],
                       population_size=vv['population_size'], num_elites=vv['num_elites'], kwargs=vv)

    with open(vv['two_arms_expert_data'], 'rb') as f:
        two_arms_data = pickle.load(f)
        two_arms_data_configs = two_arms_data['configs']
        two_arms_data_state_trajs = two_arms_data['state_trajs']
        if vv['fwd_dyn_mode'] == 'reduced_obs' or vv['fwd_dyn_mode'] == 'state':
            two_arms_data_goal_obs = two_arms_data['ob_next_trajs']
        elif vv['fwd_dyn_mode'] == 'particles':
            two_arms_data_goal_obs = two_arms_data['particles_next_trajs']

    if vv['is_eval'] == False and 'wandb' in vv and vv['wandb']:
        wandb_run = wandb.init(
            project="deformable-soil",
            entity="ctorl",
            config=vv,
            name=vv['name'],
        )
    else:
        wandb_run = None

    if vv['is_eval']:
        assert vv['cem_traj'] is not None
        with open(vv['cem_traj'], 'rb') as f:
            cem_traj = pickle.load(f)

        indices_for_playback = vv['indices_for_playback']
        if indices_for_playback is not None:
            indices = cem_traj['two_arms_data_indices']
        else:
            indices = [vv['index_for_playback']]

        total_normalized_perf_final = []
        total_normalized_perf_final_first_100eps = []
        total_normalized_perf_final_gt20 = []
        total_normalized_perf_final_gt30 = []
        ckpt_folder = '/'.join(vv['cem_traj'].split('/')[:-1])

        configs = []
        state_trajs = []
        action_trajs = []
        ob_trajs = []
        ob_next_trajs = []
        reward_trajs = []
        done_trajs = []
        ob_img_trajs = []
        ob_img_next_trajs = []
        saved_indices = []
        for i, target_ep_index in enumerate(indices):
            if vv['only_save_num_eps'] is not None and vv['only_save_num_eps'] == len(saved_indices):
                break

            # reset environment
            env.reset()
            if isinstance(two_arms_data_state_trajs[target_ep_index, 0], str):
                env.set_scene(two_arms_data_configs[target_ep_index], np.load(two_arms_data_state_trajs[target_ep_index, 0], allow_pickle=True).item())
            else:
                env.set_scene(two_arms_data_configs[target_ep_index], two_arms_data_state_trajs[target_ep_index, 0])
            obs = env._reset()

            env.start_record()

            # from PIL import Image; im = Image.fromarray(env.get_image(720, 720)); im.save("debug_env.jpeg") # for debugging

            action_trajs_cem = cem_traj['action_trajs'][i]
            infos = []
            states = []
            actions = []
            observations = []
            next_observations = []
            rewards = []
            dones = []
            ob_img_traj = []
            next_ob_img_traj = []
            for act in action_trajs_cem:
                states.append(env.get_state())
                actions.append(act)
                if torch.is_tensor(obs):
                    observations.append(obs.clone().detach().cpu().numpy())
                else:
                    observations.append(obs)
                ob_img_traj.append(env.get_image(vv['env_img_size'], vv['env_img_size']))

                obs, reward, done, info = env.step(act)
                infos.append(info)
                cur_img = env.get_image(vv['env_img_size'], vv['env_img_size'])
                next_ob_img_traj.append(cur_img)
                next_obs = obs
                if torch.is_tensor(next_obs):
                    next_observations.append(next_obs.clone().detach().cpu().numpy())
                else:
                    next_observations.append(next_obs)
                rewards.append(reward)
                dones.append(done)

            normalized_perf_final = infos[-1]['normalized_performance']
            print('info_normalized_performance_final: ', normalized_perf_final)
            total_normalized_perf_final.append(normalized_perf_final)

            if normalized_perf_final > 0.2:
                total_normalized_perf_final_gt20.append(normalized_perf_final)
            if normalized_perf_final > 0.3:
                total_normalized_perf_final_gt30.append(normalized_perf_final)
            if len(total_normalized_perf_final_first_100eps) < 100:
                total_normalized_perf_final_first_100eps.append(normalized_perf_final)

            env.end_record(video_path=os.path.join(ckpt_folder, f'ep_num_{target_ep_index}_{normalized_perf_final}.gif'))

            # only save transition if greater than 30% performance
            if normalized_perf_final > 0.3:
                configs.append(env.get_current_config().copy())
                state_trajs.append(states)
                action_trajs.append(actions)
                ob_trajs.append(observations)
                ob_next_trajs.append(next_observations)
                reward_trajs.append(rewards)
                done_trajs.append(dones)
                ob_img_trajs.append(np.array(ob_img_traj.copy()))
                ob_img_next_trajs.append(np.array(next_ob_img_traj.copy()))
                saved_indices.append(target_ep_index)

        print(f'average normalized_performance_final: {np.mean(total_normalized_perf_final)}')
        print(f'standard deviations: {np.std(total_normalized_perf_final)}')

        print(f'\nNumber of episodes greater than 20%: {len(total_normalized_perf_final_gt20)}')
        print(f'average normalized_performance_final of eps greater than 20%: {np.mean(total_normalized_perf_final_gt20)}')
        print(f'standard deviations of eps greater than 20%: {np.std(total_normalized_perf_final_gt20)}')

        print(f'\nNumber of episodes greater than 30%: {len(total_normalized_perf_final_gt30)}')
        print(f'average normalized_performance_final of eps greater than 30%: {np.mean(total_normalized_perf_final_gt30)}')
        print(f'standard deviations of eps greater than 30%: {np.std(total_normalized_perf_final_gt30)}')

        print(f'\nNumber of episodes for the first 100 episodes: {len(total_normalized_perf_final_first_100eps)}')
        print(f'average normalized_performance_final of the first 100 episodes: {np.mean(total_normalized_perf_final_first_100eps)}')
        print(f'standard deviations of the first 100 episodes: {np.std(total_normalized_perf_final_first_100eps)}')

        if vv['save_dataset']:
            traj_data = dict(configs=configs,
                state_trajs=np.array(state_trajs),
                action_trajs=np.array(action_trajs),
                ob_trajs=np.array(ob_trajs),
                ob_next_trajs=np.array(ob_next_trajs),
                ob_img_trajs=np.array(ob_img_trajs),
                ob_img_next_trajs=np.array(ob_img_next_trajs),
                reward_trajs=np.array(reward_trajs),
                done_trajs=np.array(done_trajs),
                saved_indices=np.array(saved_indices),
                total_normalized_perf_final_first_100eps=np.array(total_normalized_perf_final_first_100eps),
                env_kwargs=env_kwargs)
            out_file_path = os.path.join(ckpt_folder, f'one_arm_dataset_{len(saved_indices)}eps.pkl')
            with open(out_file_path, 'wb') as fh:
                pickle.dump(traj_data, fh, protocol=4)
            print(f'{out_file_path} saved!!!')
    else:
        # Run policy
        initial_states, state_trajs, action_trajs, ob_trajs, ob_img_trajs, configs, all_infos, reward_trajs, done_trajs, ob_next_trajs = [], [], [], [], [], [], [], [], [], []
        teacher_data_num_eps = vv['teacher_data_num_eps']
        if teacher_data_num_eps is not None:
            if teacher_data_num_eps == len(two_arms_data_configs):
                # iterate all episodes in the two arms teacher dataset
                indices = [i for i in range(len(two_arms_data_configs))]
            else:
                lst = range(0, len(two_arms_data_configs))
                indices = random.choices(lst, k=teacher_data_num_eps) # no repeat random numbers between 0 and len(two_arms_data)
        else:
            indices = [vv['teacher_data_ep_index']]

        avg_normalized_perf_final, avg_rewards = [], []
        total_ep_num_fdy_model_interactions = 0
        for index, ep_num in enumerate(indices):
            logger.log('teacher dataset episode ' + str(ep_num))
            # environment reset
            env.reset()
            if isinstance(two_arms_data_state_trajs[ep_num, 0], str):
                env.set_scene(two_arms_data_configs[ep_num], np.load(two_arms_data_state_trajs[ep_num, 0], allow_pickle=True).item())
            else:
                env.set_scene(two_arms_data_configs[ep_num], two_arms_data_state_trajs[ep_num, 0])
            obs = env._reset()

            policy.reset()
            initial_state = env.get_state()

            state_traj = []
            action_traj = []
            ob_traj = []
            next_ob_traj = []
            infos = []
            total_reward = 0
            rewards = []
            dones = []
            per_ep_num_fdy_model_interactions = 0
            for j in range(env.horizon):
                logger.log('episode {}, step {}'.format(ep_num, j))
                action, num_fdy_model_interactions = policy.get_action(obs, two_arms_data_goal_obs[ep_num, 0], j)
                action_traj.append(copy.copy(action))
                if torch.is_tensor(obs):
                    ob_traj.append(copy.copy(obs.clone().detach().cpu().numpy()))
                else:
                    ob_traj.append(copy.copy(obs))
                state_traj.append(env.get_state())
                obs, reward, done, info = env.step(action)
                infos.append(info)
                rewards.append(reward)
                total_reward += reward
                dones.append(done)
                if torch.is_tensor(obs):
                    next_ob_traj.append(copy.copy(obs.clone().detach().cpu().numpy()))
                else:
                    next_ob_traj.append(copy.copy(obs))
                per_ep_num_fdy_model_interactions += num_fdy_model_interactions

            all_infos.append(infos)
            initial_states.append(initial_state.copy())
            state_trajs.append(state_traj)
            action_trajs.append(action_traj.copy())
            ob_trajs.append(ob_traj.copy())
            configs.append(env.get_current_config().copy())
            reward_trajs.append(rewards.copy())
            done_trajs.append(dones.copy())
            ob_next_trajs.append(next_ob_traj.copy())

            normalized_performance_final = infos[-1]['normalized_performance']
            avg_rew = int(total_reward / env.horizon)
            if wandb_run:
                wandb_log_dict = {
                    "Episode": index,
                    "val/info_normalized_performance_final": normalized_performance_final,
                    "val/avg_rews": avg_rew,
                    "val/avg_ep_length": env.horizon,
                }
                wandb.log(wandb_log_dict)
            else:
                print(f'val/info_normalized_performance_final: {normalized_performance_final}')
                print(f'val/avg_rews: {avg_rew}')
                print(f'val/avg_ep_length: {env.horizon}')
                print(f'Forward dynamics model interactions in this episode: {per_ep_num_fdy_model_interactions}')

            total_ep_num_fdy_model_interactions += per_ep_num_fdy_model_interactions
            avg_normalized_perf_final.append(normalized_performance_final)
            avg_rewards.append(avg_rew)

        # Dump trajectories
        traj_dict = {
            'configs': configs,
            'initial_states': initial_states,
            'state_trajs': np.array(state_trajs),
            'action_trajs': np.array(action_trajs),
            'ob_trajs': np.array(ob_trajs),
            'ob_next_trajs': np.array(ob_next_trajs),
            'reward_trajs': np.array(reward_trajs),
            'done_trajs': np.array(done_trajs),
            'two_arms_data_indices': np.array(indices),
            'total_ep_num_fdy_model_interactions': total_ep_num_fdy_model_interactions,
        }
        with open(osp.join(log_dir, f'cem_traj.pkl'), 'wb') as cem_traj_file_handle:
            pickle.dump(traj_dict, cem_traj_file_handle)

        # Dump video
        # cem_make_gif(env_render, initial_states, action_trajs, configs, logger.get_dir(), vv['env_name'] + '.gif')

        avg_normalized_perf_final = np.average(avg_normalized_perf_final)
        avg_rewards = np.average(avg_rewards)
        if wandb_run:
            wandb_log_dict = {
                "val/avg_info_normalized_performance_final": avg_normalized_perf_final,
                "val/total_avg_rews": avg_rewards,
            }
            wandb.log(wandb_log_dict)
            wandb_run.finish()
        else:
            print(f'val/avg_info_normalized_performance_final: {avg_normalized_perf_final}')
            print(f'val/total_avg_rews: {avg_rewards}')
            print(f'Total number of forward dynamics model interactions in {len(indices)} episodes: {total_ep_num_fdy_model_interactions}')

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Evaluation
    parser.add_argument('--is_eval',  default=False, type=str2bool, help="evaluation or training mode")
    parser.add_argument('--cem_traj', default=None, type=str, help="file path to cem_traj.pkl (contains action_trajs for playback)")
    parser.add_argument('--index_for_playback', default=0, type=int, help="index to two_arms_expert for playback")
    parser.add_argument('--indices_for_playback',  default=False, type=str2bool, help="whether to load indices for playback")
    parser.add_argument('--save_dataset', default=False, type=str2bool, help="Whether to save the 1 arm student dataset")
    parser.add_argument('--only_save_num_eps', default=None, type=int, help="Only save the number of specified episodes")
    parser.add_argument('--env_img_size', type=int, default=32, help='Environment (observation) image size')
    parser.add_argument('--cam_view', default='side', choices=['top_down', 'side'])

    # Experiment
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--exp_name', default='cem_2armsto1', type=str)
    parser.add_argument('--env_name', default='ClothFold')
    parser.add_argument('--log_dir', default='./data/cem_2armsto1')
    parser.add_argument('--seed', default=100, type=int)

    parser.add_argument('--teacher_data_ep_index', default=0, type=int, help="index to teacher dataset episode to compute best action trajectory ")
    parser.add_argument('--teacher_data_num_eps', default=None, type=int, help="number of episodes to compute for the teacher dataset")
    parser.add_argument('--max_iters', default=20, type=int) # default 10
    parser.add_argument('--timestep_per_decision', default=21000, type=int) # default 21000
    parser.add_argument('--use_mpc', default=True, type=bool)
    parser.add_argument('--two_arms_expert_data', default=None, type=str, help='Two arms demonstration data')
    parser.add_argument('--wandb', action='store_true', help="use wandb instead of tensorboard for logging")
    parser.add_argument('--enable_trained_fwd_dyn',  default=False, type=str2bool, help="Whether to use trained forward dynamics model to replace the simulator during CEM computations")
    parser.add_argument('--pretrained_fwd_dyn_ckpt', default=None, type=str, help="file path to pre-trained forward dynamics model's checkpoint")
    parser.add_argument('--fwd_dyn_mode', default='reduced_obs', choices=['reduced_obs', 'particles', 'state'], help='reduced_obs uses LSTM fwd. dyn. model; particles uses particle-based CNN fwd. dyn. model')
    parser.add_argument('--particle_based_cnn_lstm_fwd_dyn_mode', default='1dconv', choices=['1dconv', '2dconv'], help='Choose a particle-based forward dynamics model implementation')
    parser.add_argument('--particle_based_fwd_dyn_impl', default='cnn_lstm', choices=['cnn_lstm', 'cnn', 'lstm_particles', 'perceiverio'], help='Choose a particle-based forward dynamics model implementation')
    parser.add_argument('--enable_downsampling', default=False, type=str2bool, help="Whether to downsample cloth")
    parser.add_argument('--cem_plan_horizon', default=None, type=int)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=False, type=bool)
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
    parser.add_argument('--env_kwargs_num_variations', default=100, type=int) # default is 1000
    parser.add_argument('--env_kwargs_cam_view', default='top_down', choices=['top_down', 'side'])
    parser.add_argument('--env_kwargs_num_picker', default=1, type=int, help='Overwrite num_picker in the environment')
    parser.add_argument('--env_kwargs_action_repeat', type=int, default=1, help='Overwrite action_repeat in the environment')
    parser.add_argument('--env_kwargs_action_mode', type=str, default='pickerpickandplace', help='Overwrite action_mode in the environment')
    parser.add_argument('--env_kwargs_horizon', type=int, default=3, help='Set a non-default number of steps for the episode')

    # ThreeCubes environment
    parser.add_argument('--enable_animations', default=False, type=str2bool, help="Whether to enable animations during evaluations")

    args = parser.parse_args()

    env_name = args.env_name
    now = datetime.now().strftime("%m.%d.%H.%M")
    args.name = f'{env_name}_CEM_{now}' if not args.name else args.name
    args.log_dir=f'{args.log_dir}/{args.name}'

    run_task(args.__dict__, args.log_dir, args.exp_name)


if __name__ == '__main__':
    main()
