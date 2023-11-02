import abc
import numpy as np
import matplotlib.pyplot as plt
import os

class optim_prob(abc.ABC):
  _goal:float
  _nsteps: int
  _dim_state: int
  _dim_action: int

  @abc.abstractmethod
  def __init__(self):
    pass

  def write_to_xml(self, **kwargs):
  # write to specified xml file
  # don't use self._xml_file because it may be overwritten when using multiple envs in parallel
    env_xml_ = kwargs.get('xml_out', os.path.abspath(os.path.join("./tmp/", 'saved.xml')))
    with open(env_xml_, 'w') as f:
      self.env.sim.save(f, format='xml', keep_inertials=False)
    return env_xml_


  def get_name(self):
    raise NotImplementedError


  def get_init_state(self):
    return np.hstack([self.env.init_qpos, self.env.init_qvel]).flatten()


  def get_desired_state(self, vals:np.array, indices:np.array = None):
    x0 = self.get_init_state()
    if indices is None: # use all indices
      assert vals.shape == x0.shape
      xf = vals
    else:
      assert vals.shape == indices.shape
      assert isinstance(indices, np.ndarray) and isinstance(indices[0], np.int0)
      xf = x0
      xf[indices] = vals
    return xf


  @abc.abstractmethod
  def fitness_directTO(self):
    raise NotImplementedError


  def jac_fitness_directTO(self):
    raise NotImplementedError


  def fitness_indirectTO(self):
    raise NotImplementedError


  def jac_fitness_indirectTO(self):
    raise NotImplementedError


  # Get action bounds
  def get_action_bounds(self, scipy=False):
    upper = np.ones(self._dim_action *
                    self._nsteps)  # upper limit 1
    lower = -upper  # lower limit -1
    if scipy:
      bounds = [[lower[i], upper[i]] for i in range(len(lower))]
    else:
      bounds = (lower, upper)
    return bounds

  @abc.abstractmethod
  def rollout(self):
    raise NotImplementedError


  def save_video(self, states:np.array):
    raise NotImplementedError


  # Dynamics constraints
  def dynamics_cons(self, dv: np.array):
    sa = self.extract_states_actions_from_dv(dv)
    states = sa['states']
    actions = sa['actions']

    cons = np.empty((0,))
    for t in range(self._nsteps):
      cons = np.hstack([cons, states[t+1] - self.one_step(states[t], actions[t])])

    return np.array(cons)


  def jac_dynamics_cons(self, dv:np.array):
    raise NotImplementedError


  def jac_dynamics_cons_sparse(self, dv:np.array):
    raise NotImplementedError


  def jac_dynamics_cons_sparsity(self, dv:np.array):
    raise NotImplementedError


  def boundary_cons(self, dv:np.array):
    raise NotImplementedError


  def jac_boundary_cons_sparse(self, dv:np.array):
    raise NotImplementedError


  def jac_boundary_cons_sparsity(self):
    raise NotImplementedError


  def extract_states_actions_from_dv(self, dv: np.array):
    raise NotImplementedError


  def plot_trajectories(self, dv, init_dv, expt_folder:str="."):
    plotdir = os.path.join(expt_folder, "plots")
    os.makedirs(plotdir)

    sa = self.extract_states_actions_from_dv(dv)
    states_optim = sa['states']
    actions_optim = sa['actions']

    sa = self.extract_states_actions_from_dv(init_dv)
    states_init = sa['states']
    actions_init = sa['actions']
    if states_init.size == 0: # no states in init_dv
      rollout = self.rollout(actions_init.flatten(), save_states=True)
      states_init = rollout['states']

    rolloutx = self.rollout(actions_optim.flatten(), save_states=True)
    states_rollout = rolloutx['states']
    actions_rollout = actions_optim

    if states_optim.size > 0: # exclude indirectTO
      assert states_optim.shape == states_init.shape == states_rollout.shape # not true for multiple shooting
    assert actions_optim.shape == actions_init.shape

    t = np.arange(self._nsteps+1) * self.env.frame_skip * self.env.model.opt.timestep
    for i in range(self._dim_state):
      plt.plot(t, states_rollout[:, i])
      if states_init.size > 0:
        plt.plot(t, states_init[:,i])
      if states_optim.size > 0:
        plt.plot(t, states_optim[:,i])
      plt.legend(['rollout', 'init', 'optimized'])
      plt.xlabel('Time (s)')
      plt.title(f'state{i}')
      plt.savefig(f'{plotdir}/state{i}.png')
      plt.clf()
      plt.close()

    t = np.arange(self._nsteps) * self.env.frame_skip * self.env.model.opt.timestep  # one less action to plot
    for i in range(self._dim_action):
      plt.plot(t, actions_init[:,i])
      plt.plot(t, actions_optim[:,i])
      plt.legend(['init', 'optimized & rollout'])
      plt.xlabel('Time (s)')
      plt.title(f'action{i}')
      plt.savefig(f'{plotdir}/action{i}.png')
      plt.clf()
      plt.close()

    print(f'Done with all plots. See {plotdir}')


  def one_step(self, state, action):
    # Set state
    qpos = state[0:self.env.sim.model.nq]
    qvel = state[self.env.sim.model.nq:]
    self.env.set_state(qpos, qvel)

    # Apply action
    self.env.step(action=action)

    # Return state. Remove first element(time)
    # Valid since get_state().act and .udd_state are empty
    return self.env.sim.get_state().flatten()[1:]


  def close(self): # close env
    self.env.close()


  def set_step_height(self, step_height, stairs=False):
    self._step_height = step_height
    self.env.model.body_pos[self.step0_idx, 2] = self._step_height
    if stairs:
      self.env.model.body_pos[self.step0_idx + 1, 2] = self._step_height + 0.1
    self.env.sim.forward() # won't increment timestep by 1
    return

  def set_step_width(self, step_width):
    self._step_width = step_width
    left_edge = self.env.model.body_pos[self.step0_idx, 0] - self.env.model.geom_size[self.step0geom_idx, 0]
    self.env.model.body_pos[self.step0_idx, 0] = left_edge + self._step_width/2
    self.env.model.geom_size[self.step0geom_idx, 0] = self._step_width/2 # stores half-width for object type box
    self.env.sim.forward() # won't increment timestep by 1
    return

  '''
  Reset the env inside this optimization problem.
  Set the appropriate step dimensions (they get reset because the xml doesn't have them)
  '''
  def reset_env(self, stairs=False):
    self.env.reset()
    self.set_step_height(self._step_height, stairs=stairs)
    if self._step_width >= 0:
      self.set_step_width(self._step_width)
