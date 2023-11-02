import json
import warnings
from misc import NumpyArrayEncoder
import numpy as np
import cma
from cloth_fold_prob import ClothFoldProb
from dry_cloth_prob import DryClothProb
from optim_prob import optim_prob
import datetime
import os
import matplotlib as plt
import argparse
import wandb
from sb3.utils import str2bool
import pickle
from curl import utils
import random

def run_cma(prob:optim_prob, kwargs):
  ### PARAMS
  cma_initial_std = kwargs.get('initial_std', 0.4)
  cma_popsize = kwargs.get('popsize', 100)
  maxiter:int = kwargs.get('maxiter', 100)
  logging = kwargs.get('logging', True)
  seed = kwargs.get('seed', 3)
  # plot_initial_u = kwargs.get('plot_init_u', False)
  method_str = 'CMAES'
  prefix = kwargs.get('prefix', prob.get_name()) # Env prefix
  debug = kwargs.get('debug', False)
  enable_wandb = kwargs.get('wandb', False)

  ### SET UP LOGGER
  timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')
  run_name = f"{prefix}_{timestamp}_{method_str}"
  outdir = os.path.join(os.getcwd(), "out", run_name)
  os.makedirs(outdir)
  outpath = os.path.join(outdir, f'{timestamp}')
  print(f"Starting expt @ pid {os.getpid()}, see {outdir}")
  if debug:
    print("Force logging off because debug is on")
    logging = False

#   if logging:
#     logfile = open(f"{outpath}_stdout.log", 'w')
#   else:
#     logfile = sys.__stdout__
#   sys.stdout = logfile
#   sys.stderr = logfile

  config = {}
  config['debug'] = debug
  config['logging'] = logging
  config['cma_initial_std'] = cma_initial_std
  config['pop_size'] = cma_popsize
  config['maxiter'] = maxiter
  config['seed'] = seed
  # config['plot_initial_u'] = plot_initial_u
  config['method_str'] = method_str
  config['prefix'] = prefix
  config['nsteps'] = prob._nsteps
  config['outdir'] = outdir

  if enable_wandb:
    wandb_run = wandb.init(
        project="deformable-soil",
        entity="ctorl",
        config=config,
        name=run_name,
    )
  else:
    wandb_run = None

  # Separate RNG to be safe from global changes to numpy's RNG
  rng = np.random.default_rng(seed) # ONLY SEED ONCE PER EXPT!

  # init guess close to zero, don't bias optimizer
  init_u = rng.normal(size=prob._dim_action * prob._nsteps) * 0.01
  print(f"Initial guess computed")

  init_dv:np.array = np.array(init_u).flatten() # initial decision vector
  config['initial_fitness'] = prob.fitness_indirectTO(init_u)

  ### STORE CONFIG
  with open(f'{outpath}_config.json', 'w') as file:
    json.dump(config, file, indent=2, cls=NumpyArrayEncoder)
    print("Stored config json file")

  ### OPTIMIZE
  # boundary constraints of cma
  # http://cma.gforge.inria.fr/apidocs-pycma/cma.transformations.BoxConstraintsLinQuadTransformation.html
  tf = cma.transformations.BoxConstraintsLinQuadTransformation([[-1, 1]])
  with warnings.catch_warnings(record=True) as warns:
    os.makedirs(f'{outdir}/cma')
    es = cma.CMAEvolutionStrategy(init_dv, cma_initial_std, {'seed': seed,
                                                             'transformation': [tf.transform, tf.inverse],
                                                             'verb_filenameprefix': f'{outdir}/cma/',
                                                             'popsize': cma_popsize,
                                                             })
    # es.optimize(prob.fitness_indirectTO, args=(), iterations=maxiter, verb_disp=100, n_jobs=-1) # remove n_jobs to disable multiprocessing
    es.optimize(prob.fitness_indirectTO, args=(), iterations=maxiter, verb_disp=100) # remove n_jobs to disable multiprocessing
    es.result_pretty()
    try:
      es.logger.load(filenameprefix=f'{outdir}/cma/')
      plt.plot(es.logger.f[:, 5])
      plt.xlabel('Iterations')
      plt.ylabel('Function value')
      plt.savefig(f'{outdir}/cma_fitness.png')
      plt.close()
    except:
      print(f'Could not save data or plot to {outdir}')

  ### POST-PROCESS
  best_actions = es.best.x
  best_fitness = es.best.f
  print(f'best_fitness = {best_fitness}, best_action = {best_actions}')
  prob.rollout(best_actions, video=True)
  # info = prob.rollout(u=dv_full,
  #             save_states=True,
  #             visualize=VizOptions.VIDEO,
  #             vid_path=f'{outpath}_rollout.mp4')
  save_dv_path = f"{outpath}.npz"
  np.savez(save_dv_path,
            u=es.best.x,
            # x = info['states'].flatten(),
            fitness = es.best.f)
  # # Save data for imitation learning
  # possible_solutions = es.ask()
  # best_trajs_actions = [dv_full]
  # best_trajs_states = [info['states']]
  # best_trajs_fitness = [es.best.f]

  # prob.post_process(dv, file_prefix=f"{outpath}")
  # prob.plot_trajectories(es.best.x.flatten(), u0_rbf.flatten(), expt_folder=outdir)
  # prob.close()
  # print(f"\nEND OF deriv_free_trajopt(), pid {os.getpid()}")
  # if logging:
  #   logfile.close()
  #   sys.stdout = sys.__stdout__
  #   print(f"Expt Wrote process output to {logfile.name}")

  # return save_dv_path
  if wandb_run:
    wandb_run.finish()

def run_cma_learned_forward_dynamics_model(prob:optim_prob, kwargs):
  ### PARAMS
  cma_initial_std = kwargs.get('initial_std', 0.4)
  cma_popsize = kwargs.get('popsize', 100)
  maxiter:int = kwargs.get('maxiter', 100)
  logging = kwargs.get('logging', True)
  seed = kwargs.get('seed', 11)
  # plot_initial_u = kwargs.get('plot_init_u', False)
  method_str = 'CMAES'
  prefix = kwargs.get('prefix', prob.get_name()) # Env prefix
  debug = kwargs.get('debug', False)
  enable_wandb = kwargs.get('wandb', False)
  two_arms_expert_data = kwargs.get('two_arms_expert_data')
  assert two_arms_expert_data is not None
  teacher_data_num_eps = kwargs.get('teacher_data_num_eps')

  ### SET UP LOGGER
  timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')
  run_name = f"{prefix}_{timestamp}_{method_str}"
  outdir = os.path.join(os.getcwd(), "data", "cmaes", run_name)
  os.makedirs(outdir)
  outpath = os.path.join(outdir, f'{timestamp}')
  print(f"Starting expt @ pid {os.getpid()}, see {outdir}")
  if debug:
    print("Force logging off because debug is on")
    logging = False

  config = {}
  config['debug'] = debug
  config['logging'] = logging
  config['cma_initial_std'] = cma_initial_std
  config['pop_size'] = cma_popsize
  config['maxiter'] = maxiter
  config['seed'] = seed
  # config['plot_initial_u'] = plot_initial_u
  config['method_str'] = method_str
  config['prefix'] = prefix
  config['nsteps'] = prob._nsteps
  config['outdir'] = outdir

  if enable_wandb:
    wandb_run = wandb.init(
        project="deformable-soil",
        entity="ctorl",
        config=config,
        name=run_name,
    )
  else:
    wandb_run = None

  # load two arms teacher dataset
  with open(two_arms_expert_data, 'rb') as f:
    two_arms_data = pickle.load(f)
    two_arms_data_configs = two_arms_data['configs']
    two_arms_data_state_trajs = two_arms_data['state_trajs']
    two_arms_data_goal_obs = two_arms_data['particles_next_trajs']

  utils.set_seed_everywhere(seed)

  if teacher_data_num_eps is not None:
    if teacher_data_num_eps == len(two_arms_data_configs):
      # iterate all episodes in the two arms teacher dataset
      indices = [i for i in range(len(two_arms_data_configs))]
    else:
      lst = range(0, len(two_arms_data_configs))
      indices = random.choices(lst, k=teacher_data_num_eps) # no repeat random numbers between 0 and len(two_arms_data)
  else:
    indices = [0]

  avg_normalized_perf_final, action_trajs, fitness_trajs = [], [], []
  for index, ep_num in enumerate(indices):
    print('teacher dataset episode ' + str(ep_num))

    # construct episode data from teacher dataset
    ep_data_teacher = {
      'config': two_arms_data_configs[ep_num],
      'state': two_arms_data_state_trajs[ep_num, 0],
      'goal_obs': two_arms_data_goal_obs[ep_num, 0],
    }

    # Separate RNG to be safe from global changes to numpy's RNG
    rng = np.random.default_rng(seed) # ONLY SEED ONCE PER EXPT!

    # init guess close to zero, don't bias optimizer
    init_u = rng.normal(size=prob._dim_action * prob._nsteps) * 0.01
    print(f"Initial guess computed")

    init_dv:np.array = np.array(init_u).flatten() # initial decision vector
    config['initial_fitness'] = prob.fitness_indirectTO(init_u, ep_data_teacher)

    ### STORE CONFIG
    with open(f'{outpath}_index_{index}_ep_num_{ep_num}_config.json', 'w') as file:
      json.dump(config, file, indent=2, cls=NumpyArrayEncoder)
      print("Stored config json file")

    ### OPTIMIZE
    # boundary constraints of cma
    # http://cma.gforge.inria.fr/apidocs-pycma/cma.transformations.BoxConstraintsLinQuadTransformation.html
    tf = cma.transformations.BoxConstraintsLinQuadTransformation([[-1, 1]])
    with warnings.catch_warnings(record=True) as warns:
      if not os.path.exists(f'{outdir}/cma'):
        os.makedirs(f'{outdir}/cma')
      es = cma.CMAEvolutionStrategy(init_dv, cma_initial_std, {'seed': seed,
                                                              'transformation': [tf.transform, tf.inverse],
                                                              'verb_filenameprefix': f'{outdir}/cma/',
                                                              'popsize': cma_popsize,
                                                              })
      es.optimize(prob.fitness_indirectTO, args=(), iterations=maxiter, verb_disp=100) # remove n_jobs to disable multiprocessing
      es.result_pretty()

    ### POST-PROCESS
    best_actions = es.best.x
    best_fitness = es.best.f
    print(f'best_fitness = {best_fitness}, best_action = {best_actions}')
    action_trajs.append(best_actions)
    fitness_trajs.append(best_fitness)
    normalized_perf_final = prob.rollout_for_stats_and_video(best_actions, ep_data_teacher, index, ep_num, outdir)
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
    'fitness_trajs': np.array(fitness_trajs),
    'two_arms_data_indices': np.array(indices),
    'total_normalized_perf_final_first_100eps': np.array(avg_normalized_perf_final),
  }
  with open(os.path.join(outdir, f'cmaes_traj.pkl'), 'wb') as file_handle:
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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--wandb', action='store_true', help="use wandb instead of tensorboard for logging")
  parser.add_argument('--teacher_data_num_eps', default=None, type=int, help="number of episodes to compute for the teacher dataset")
  parser.add_argument('--two_arms_expert_data', default=None, type=str, help='Two arms demonstration data')
  parser.add_argument('--enable_trained_fwd_dyn',  default=False, type=str2bool, help="Whether to use trained forward dynamics model to replace the simulator during CMA-ES computations")
  parser.add_argument('--pretrained_fwd_dyn_ckpt', default=None, type=str, help="file path to pre-trained forward dynamics model's checkpoint")
  parser.add_argument('--seed', default=11, type=int, help="seed number")

  args = parser.parse_args()

  # prob = ClothFoldProb(args.__dict__)
  prob = DryClothProb(args.__dict__)

  if args.enable_trained_fwd_dyn:
    run_cma_learned_forward_dynamics_model(prob, args.__dict__)
  else:
    run_cma(prob, args.__dict__)
