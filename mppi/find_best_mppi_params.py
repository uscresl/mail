
import os
import argparse
from sb3.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=11, type=int, help="seed number")
parser.add_argument('--teacher_data_num_eps', default=None, type=int, help="number of episodes to compute for the teacher dataset")
parser.add_argument('--two_arms_expert_data', default=None, type=str, help='Two arms demonstration data')
parser.add_argument('--pretrained_fwd_dyn_ckpt', default=None, type=str, help="file path to pre-trained forward dynamics model's checkpoint")
args = parser.parse_args()

param_combinations = [
    ############## Set of Parameters in CoRL DryCloth Experiments  ##############
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 6000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 10000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 15000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 20000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 30000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 60000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 80000,
    },
    {
        'mppi_timesteps': 3,
        'mppi_num_samples': 120000,
    },




    ############## Parameters Tried But Didn't Work Well  ##############
    # {
    #     'mppi_timesteps': 1,
    #     'mppi_num_samples': 1000,
    # },
    # {
    #     'mppi_timesteps': 1,
    #     'mppi_num_samples': 4000,
    # },
    # {
    #     'mppi_timesteps': 1,
    #     'mppi_num_samples': 6000,
    # },
    # {
    #     'mppi_timesteps': 4,
    #     'mppi_num_samples': 6000,
    # },
    # {
    #     'mppi_timesteps': 6,
    #     'mppi_num_samples': 1000,
    # },
    # {
    #     'mppi_timesteps': 6,
    #     'mppi_num_samples': 4000,
    # },
    # {
    #     'mppi_timesteps': 6,
    #     'mppi_num_samples': 6000,
    # },
    # {
    #     'mppi_timesteps': 10,
    #     'mppi_num_samples': 1000,
    # },
    # {
    #     'mppi_timesteps': 10,
    #     'mppi_num_samples': 4000,
    # },
    # {
    #     'mppi_timesteps': 10,
    #     'mppi_num_samples': 6000,
    # },
]

for param in param_combinations:
    mppi_timesteps = param['mppi_timesteps']
    mppi_num_samples = param['mppi_num_samples']

    command = 'python mppi/mppi.py ' + \
            f'--two_arms_expert_data={args.two_arms_expert_data} ' + \
            f'--teacher_data_num_eps={args.teacher_data_num_eps} ' + \
            f'--pretrained_fwd_dyn_ckpt={args.pretrained_fwd_dyn_ckpt} ' + \
            f'--seed={args.seed} ' + \
            f'--mppi_timesteps={mppi_timesteps} ' + \
            f'--mppi_num_samples={mppi_num_samples} '

    os.system(command)