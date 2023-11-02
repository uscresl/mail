
import os 
import argparse
from sb3.utils import str2bool

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', default='ClothFold')
parser.add_argument('--env_kwargs_horizon', type=int, default=3, help='Set a non-default number of steps for the episode')
parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)
parser.add_argument('--env_kwargs_action_mode', type=str, default='pickerpickandplace', help='Overwrite action_mode in the environment')
parser.add_argument('--env_kwargs_num_picker', default=1, type=int, help='Overwrite num_picker in the environment')
parser.add_argument('--enable_trained_fwd_dyn',  default=True, type=str2bool, help="Whether to use trained forward dynamics model to replace the simulator during CEM computations")
parser.add_argument('--pretrained_fwd_dyn_ckpt', default=None, type=str, help="file path to pre-trained forward dynamics model's checkpoint")
parser.add_argument('--particle_based_cnn_lstm_fwd_dyn_mode', default='1dconv', choices=['1dconv', '2dconv'], help='Choose a particle-based forward dynamics model implementation')
parser.add_argument('--fwd_dyn_mode', default='reduced_obs', choices=['reduced_obs', 'particles', 'state'], help='reduced_obs uses LSTM fwd. dyn. model; particles uses particle-based CNN fwd. dyn. model')
parser.add_argument('--particle_based_fwd_dyn_impl', default='cnn_lstm', choices=['cnn_lstm', 'cnn', 'lstm_particles', 'perceiverio'], help='Choose a particle-based forward dynamics model implementation')
parser.add_argument('--enable_downsampling', default=False, type=str2bool, help="whether to downsample cloth")
parser.add_argument('--two_arms_expert_data', default='data/ClothFold_DynModel_Particles_1000eps_top_down_pickerpickandplace_2arm.pkl', type=str, help='Two arms demonstration data')
parser.add_argument('--num_eps', type=int, default=10, help='Number of episodes for evaluation')
parser.add_argument('--seed', type=int, default=10, help='Seed number')

args = parser.parse_args()

env_name = args.env_name
env_kwargs_horizon = args.env_kwargs_horizon
env_kwargs_observation_mode = args.env_kwargs_observation_mode
env_kwargs_action_mode = args.env_kwargs_action_mode
env_kwargs_num_picker = args.env_kwargs_num_picker
enable_trained_fwd_dyn = args.enable_trained_fwd_dyn
pretrained_fwd_dyn_ckpt = args.pretrained_fwd_dyn_ckpt
particle_based_cnn_lstm_fwd_dyn_mode = args.particle_based_cnn_lstm_fwd_dyn_mode
fwd_dyn_mode = args.fwd_dyn_mode
particle_based_fwd_dyn_impl = args.particle_based_fwd_dyn_impl
enable_downsampling = args.enable_downsampling
two_arms_expert_data = args.two_arms_expert_data
num_eps = args.num_eps
seed = args.seed

param_combinations = [
    ############## Set of Parameters in RSS DryCloth Experiments  ##############
    {
        'cem_plan_horizon': 1,
        'max_iters': 2,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 2,
        'timestep_per_decision': 15000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 2,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 2,
        'timestep_per_decision': 31000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 2,
        'timestep_per_decision': 34000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 10,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 1,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 1,
        'timestep_per_decision': 15000, # default
    },
    {
        'cem_plan_horizon': 2,
        'max_iters': 1,
        'timestep_per_decision': 32000, # default
    },
    {
        'cem_plan_horizon': 3,
        'max_iters': 2,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 3,
        'max_iters': 10,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 4,
        'max_iters': 2,
        'timestep_per_decision': 21000, # default
    },
    {
        'cem_plan_horizon': 4,
        'max_iters': 10,
        'timestep_per_decision': 21000, # default
    },




    ############## ClothFold ##############
    # run this on GPU 0
    # {
    #     'cem_plan_horizon': 1,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 1,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 1,
    #     'timestep_per_decision': 31000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 1,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 1,
    #     'timestep_per_decision': 16000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 1,
    #     'timestep_per_decision': 11000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 1,
    #     'timestep_per_decision': 9000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 18000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 25000,
    # },

    # run this on GPU 1
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 16000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 11000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 20,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 30,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 3,
    #     'max_iters': 1,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 3,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 3,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000,
    # },
    # {
    #     'cem_plan_horizon': 6,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000,
    # },

    # additional parameters to run


    ############## DryCloth ##############
    # run this on GPU 0
    # {
    #     'cem_plan_horizon': 1,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 2,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 3,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 3,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 4,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 4,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },


    # run this on GPU 1
    # {
    #     'cem_plan_horizon': 6,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 6,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 7,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 7,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 9,
    #     'max_iters': 2,
    #     'timestep_per_decision': 21000, # default
    # },
    # {
    #     'cem_plan_horizon': 9,
    #     'max_iters': 10,
    #     'timestep_per_decision': 21000, # default
    # },
]

for param in param_combinations:
    cem_plan_horizon = param['cem_plan_horizon']
    max_iters = param['max_iters']
    timestep_per_decision = param['timestep_per_decision']

    command = 'python experiments/run_cem_2armsto1.py ' + \
            '--name=DEBUG ' + \
            f'--env_name={env_name} ' + \
            f'--max_iters={max_iters} ' + \
            f'--env_kwargs_horizon={env_kwargs_horizon} ' + \
            f'--env_kwargs_observation_mode={env_kwargs_observation_mode} ' + \
            f'--env_kwargs_action_mode={env_kwargs_action_mode} ' + \
            f'--env_kwargs_num_picker={env_kwargs_num_picker} ' + \
            f'--two_arms_expert_data={two_arms_expert_data} ' + \
            f'--teacher_data_num_eps={num_eps} ' + \
            f'--pretrained_fwd_dyn_ckpt={pretrained_fwd_dyn_ckpt} ' + \
            f'--enable_trained_fwd_dyn={enable_trained_fwd_dyn} ' + \
            f'--fwd_dyn_mode={fwd_dyn_mode} ' + \
            f'--particle_based_cnn_lstm_fwd_dyn_mode={particle_based_cnn_lstm_fwd_dyn_mode} ' + \
            f'--cem_plan_horizon={cem_plan_horizon} ' + \
            f'--timestep_per_decision={timestep_per_decision} ' + \
            f'--particle_based_fwd_dyn_impl={particle_based_fwd_dyn_impl} ' + \
            f'--seed={seed} ' + \
            f'--enable_downsampling={enable_downsampling}'

    os.system(command)