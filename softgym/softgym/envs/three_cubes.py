import numpy as np
from gym.spaces import Box
from gym import spaces

import pyflex
from softgym.envs.fluid_env import FluidEnv
import copy
from softgym.utils.misc import quatFromAxisAngle
import time

class ThreeCubesEnv(FluidEnv):
    def __init__(self, observation_mode, action_mode, cached_states_path='three_cubes_init_states.pkl', **kwargs):
        '''
        Observation Space:
            * 4 values: cube1_x, cube2_x, cube3_x, timestep
            * time horizon: t1 = 0, t2 = 0.5, t3 = 1.0

        3 actuators (three arms):
            * action space has 3 values: first controls cube1_x, second controls cube2_x, third controls cube3_x.

        1 actuator (one arm):
            * action space has 1 value: Depending on the timestep,
              we perform action on cube1, cube2, or cube3 (e.g., 0=cube1, 1=cube2, 2=cube3)

        Note that the first timestep actually does nothing to the boxes / environment (self.timestep == -1).
        '''
        assert observation_mode in ['state', 'cam_rgb_key_point']
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self.env_pyflex = pyflex
        self.num_picker = kwargs.get('num_picker', 1)
        self.env_image_size = kwargs.get('env_image_size', 32)
        self.cube_width_height_len = 0.05 # size of cube
        self.num_cubes = 3

        # starting locations
        self.cube1_x = 0
        self.cube2_x = 0.3
        self.cube3_x = 0.8
        self.cube_y_coord = 0.05

        # customized widths
        self.cube2_width = 0.08
        self.cube3_width = 0.02

        # goal locations
        self.cube1_x_goal = 1.1
        self.cube2_x_goal = 0.7
        self.cube3_x_goal = 0.1

        # timestep tracking
        self.timestep = 0
        self.timestep_onetenth = 0

        # animations
        self.enable_animations = kwargs.get('enable_animations', False)
        self.time_delta_steps = 50
        self.time_sleep = 0.01
        self.cube1_dh = 0.007
        self.cube2_dh = 0.001
        self.cube3_dh = 0.004

        super().__init__(**kwargs)

        if observation_mode == 'state':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        elif observation_mode == 'cam_rgb_key_point':
            self.observation_space = spaces.Dict(
                dict(
                    image = Box(low=-np.inf, high=np.inf, shape=(self.env_image_size, self.env_image_size, 3), dtype=np.float32),
                    key_point = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
                )
            )
        else:
            raise NotImplementedError

        if action_mode == 'direct': # control the movement of the cup
            if self.num_picker == 1:
                action_low = np.array([-0.1])  # left-most edge (with full visibility of the box)
                action_high = np.array([1.35]) # right-most edge (with full visibility of the box)
            elif self.num_picker == 2:
                action_low = np.array([-0.1, -0.1])  # left-most edge (with full visibility of the box)
                action_high = np.array([1.35, 1.35]) # right-most edge (with full visibility of the box)
            elif self.num_picker == 3:
                action_low = np.array([-0.1, -0.1, -0.1])  # left-most edge (with full visibility of the box)
                action_high = np.array([1.35, 1.35, 1.35]) # right-most edge (with full visibility of the box)
            self.action_space = Box(action_low, action_high, dtype=np.float32)
        else:
            raise NotImplementedError

        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def get_default_config(self):
        config = {
            'fluid': {
                'radius': 0,
                'rest_dis_coef': 0,
                'cohesion': 0,
                'viscosity': 0,
                'surfaceTension': 0.,
                'adhesion': 0.0,
                'vorticityConfinement': 0,
                'solidpressure': 0.,
                'dim_x': 0,
                'dim_y': 0,
                'dim_z': 0,
            },
            'glass': {
                'border': self.cube_width_height_len, # how big the box will be!!!!!
                'height': 0.0, # this won't be used, will be overwritten by generating variation
            },
            'camera_name': 'default_camera',
        }
        return config

    def generate_env_variation(self, num_variations=5, **kwargs):
        self.cached_configs = []
        self.cached_init_states = []

        config = self.get_default_config()
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]

        for idx in range(num_variations):
            config_variations[idx]['fluid']['dim_x'] = 1
            config_variations[idx]['fluid']['dim_y'] = 1
            config_variations[idx]['fluid']['dim_z'] = 1

            config_variations[idx]['glass']['height'] = config['glass']['border']

            self.set_scene(config_variations[idx])

            # randomize where cube1, cube2, and cube3 will spawn
            if self.num_picker == 3:
                cube1_rand_x = np.random.uniform(0, 0.1)
                cube2_rand_x = np.random.uniform(0.3, 0.4)
                cube3_rand_x = np.random.uniform(0.8, 0.9)
                action = np.array([cube1_rand_x, cube2_rand_x, cube3_rand_x])
                self.move_cube(self.glass_states, action)
                self.move_cube(self.glass_states, action)
                pyflex.set_shape_states(self.glass_states)
            else:
                print('!!!!! No environment variations !!!!! Please use 3 actuators to generate cache_initial_states with variations!')

            init_state = copy.deepcopy(self.get_state())

            self.cached_configs.append(config_variations[idx])
            self.cached_init_states.append(init_state)
            # print('Env variations: ', self.get_state()['shape_pos']) # for debugging

        return self.cached_configs, self.cached_init_states

    def get_config(self):
        if self.deterministic:
            config_idx = 0
        else:
            config_idx = np.random.randint(len(self.config_variations))

        self.config = self.config_variations[config_idx]
        return self.config

    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        self.timestep = self.timestep_onetenth = 0
        self.inner_step = 0
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        # pyflex.step(render=True) # original code: seg fault in CEM. Not using render=True still works fine with and without GUI
        pyflex.step()
        # make sure we start the cubes on the ground rather than under it
        states = self.glass_states
        states[0][1] = self.cube_y_coord
        states[0][4] = self.cube_y_coord
        states[1][1] = self.cube_y_coord
        states[1][4] = self.cube_y_coord
        states[2][1] = self.cube_y_coord
        states[2][4] = self.cube_y_coord
        self.glass_states = states
        pyflex.set_shape_states(self.glass_states)

        return self._get_obs()

    def get_shape_states(self):
        return pyflex.get_shape_states()

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_states': self.glass_states, 'glass_params': self.glass_params, 'config_id': self.current_config_id,
                'cube1_x': self.cube1_x, 'cube2_x': self.cube2_x, 'cube3_x': self.cube3_x, 'timestep': self.timestep, 'timestep_onetenth': self.timestep_onetenth}

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        self.glass_params = state_dic['glass_params']
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.cube1_x = state_dic['cube1_x']
        self.cube2_x = state_dic['cube2_x']
        self.cube3_x = state_dic['cube3_x']
        self.glass_states = state_dic['glass_states']
        self.timestep = state_dic['timestep']
        self.timestep_onetenth = state_dic['timestep_onetenth']
        for _ in range(5):
            pyflex.step()

    def initialize_camera(self):
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        self.camera_params = {
            # use in the paper (sim)!!!
            # 'default_camera': {'pos': np.array([0.64, 0.01, 2.0]),
            #                 'angle': np.array([0, 0, 0]),
            # use on Panda arm (real)!!!
            'default_camera': {'pos': np.array([0.5, 2.3, 0]),
                            'angle': np.array([0, -90 / 180. * np.pi, 0.]),
                            'width': self.camera_width,
                            'height': self.camera_height},
            'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                    'angle': np.array([0, 0, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'left_side': {'pos': np.array([-1, .2, 0]),
                    'angle': np.array([-0.5 * np.pi, 0, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'right_side': {'pos': np.array([2, .2, 0]),
                    'angle': np.array([0.5 * np.pi, 0, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height}
        }

    def set_glass_params(self, config=None):
        params = config

        self.border = params['border']
        self.height = params['height']
        self.glass_params = params

    def set_scene(self, config, states=None):
        '''
        Construct the passing water scence.
        '''
        # create fluid
        super().set_scene(config)  # do not sample fluid parameters, as it's very likely to generate very strange fluid

        # compute glass params
        if states is None:
            self.set_glass_params(config["glass"])
        else:
            glass_params = states['glass_params']
            self.border = glass_params['border']
            self.height = glass_params['height']
            self.glass_params = glass_params

        # create glass
        self.create_glass(self.border)

        # move glass to be at ground or on the table
        self.glass_states = self.init_glass_state()

        pyflex.set_shape_states(self.glass_states)

        if states is not None:
            # set to passed-in cached init states
            self.set_state(states)

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'state':
            return np.array([self.cube1_x, self.cube2_x, self.cube3_x, self.timestep_onetenth])
        elif self.observation_mode == 'cam_rgb_key_point':
            # DEBUG: get_image
            # sample_img = self.get_image(self.env_image_size, self.env_image_size)
            # import cv2
            # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('data/three_cubes_image.png', sample_img)
            # import pdb; pdb.set_trace()
            pos = {
                'image': self.get_image(self.env_image_size, self.env_image_size),
                'key_point': np.array([self.cube1_x, self.cube2_x, self.cube3_x, self.timestep_onetenth]),
            }
            return pos
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """
        Distance between each cube's current position and goal position
        """
        cube1_dist = np.linalg.norm(self.cube1_x - self.cube1_x_goal)
        cube2_dist = np.linalg.norm(self.cube2_x - self.cube2_x_goal)
        cube3_dist = np.linalg.norm(self.cube3_x - self.cube3_x_goal)
        reward = -(cube1_dist + cube2_dist + cube3_dist)
        return reward

    def _get_info(self):
        performance = self.compute_reward()
        performance_init = performance if self.performance_init is None else self.performance_init 
        normalized_performance = (performance - performance_init) / (0. - performance_init)
        return {'performance': performance,
                'normalized_performance': normalized_performance}

    def _step(self, action):
        '''
        action: np.ndarray of dim 1x1, dx, which specifies how much to move on the x-axis.
        '''
        if self.enable_animations:
            self.move_cube_with_animation(self.glass_states, action)
        else:
            # neede to execute two times to actually move a cube (following the original PassWater's implementation)
            self.move_cube(self.glass_states, action)
            self.move_cube(self.glass_states, action)

        # print(f'Before rendering: {self.timestep}: {self.glass_states}') # for debugging
        # pyflex takes a step to update the glass and the water fluid
        pyflex.set_shape_states(self.glass_states)
        # pyflex.step(render=True) # original code: seg fault in CEM. Not using render=True still works fine with and without GUI
        pyflex.step()

        # print(f'obs: {self._get_obs()}') # for debugging
        self.inner_step += 1
        self.timestep += 1
        self.timestep_onetenth += 0.1

    def create_glass(self, border):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        each wall of the real box is represented by a box object in Flex with really small thickness (determined by the param border)
        dis_x: the length of the glass
        dis_z: the width of the glass
        height: the height of the glass.
        border: the thickness of the glass wall.

        the halfEdge determines the center point of each wall.
        Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
        That's why left and right walls have exactly the same params, and so do front and back walls.   
        """
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # cube 1
        halfEdge = np.array([border, border, border])
        boxes.append([halfEdge, center, quat])

        # cube 2
        halfEdge = np.array([self.cube2_width, border, border])
        boxes.append([halfEdge, center, quat])

        # cube 3
        halfEdge = np.array([self.cube3_width, border, border])
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)
            pyflex.set_box_color([0.1, 0.5, 0.3])

        return boxes

    def move_cube_with_animation(self, prev_states, action):

        def _update_current_states_with_prev_states(states):
            # update current x,y,z with previous x,y,z
            states[0][:3] = states[0][3:6]
            states[0][6:10] = states[0][10:]
            states[1][:3] = states[1][3:6]
            states[1][6:10] = states[1][10:]
            states[2][:3] = states[2][3:6]
            states[2][6:10] = states[2][10:]

            # make sure we start the cubes on the ground rather than under it
            states[0][1] = self.cube_y_coord
            states[0][4] = self.cube_y_coord
            states[1][1] = self.cube_y_coord
            states[1][4] = self.cube_y_coord
            states[2][1] = self.cube_y_coord
            states[2][4] = self.cube_y_coord

        def _move_cube_with_animation(cube_index, cube_dh, states, act_clipped):
            # move up in the air
            for i in range(self.time_delta_steps):
                states[cube_index, :3] = np.array([states[cube_index, 0], states[cube_index, 1] + cube_dh, 0.])
                states[cube_index, 3:6] = np.array([states[cube_index, 0], states[cube_index, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move toward the goal location
            dx = (act_clipped - states[cube_index, 0]) / self.time_delta_steps
            for i in range(self.time_delta_steps):
                states[cube_index, :3] = np.array([states[cube_index, 0] + dx, states[cube_index, 1], 0.])
                states[cube_index, 3:6] = np.array([states[cube_index, 0], states[cube_index, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move down to the ground
            for i in range(self.time_delta_steps):
                states[cube_index, :3] = np.array([states[cube_index, 0], states[cube_index, 1] - cube_dh, 0.])
                states[cube_index, 3:6] = np.array([states[cube_index, 0], states[cube_index, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))
            return

        def _move_two_cubes_with_animation(states, act_clipped):
             # move up in the air
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0], states[0, 1] + self.cube1_dh, 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0], states[1, 1] + self.cube2_dh, 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move toward the goal location
            dx1 = (act_clipped[0] - states[0, 0]) / self.time_delta_steps
            dx2 = (act_clipped[1] - states[1, 0]) / self.time_delta_steps
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0] + dx1, states[0, 1], 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0] + dx2, states[1, 1], 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move down to the ground
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0], states[0, 1] - self.cube1_dh, 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0], states[1, 1] - self.cube2_dh, 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))
            return

        def _move_three_cubes_with_animation(states, act_clipped):
             # move up in the air
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0], states[0, 1] + self.cube1_dh, 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0], states[1, 1] + self.cube2_dh, 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[2, :3] = np.array([states[2, 0], states[2, 1] + self.cube3_dh, 0.])
                states[2, 3:6] = np.array([states[2, 0], states[2, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move toward the goal location
            dx1 = (act_clipped[0] - states[0, 0]) / self.time_delta_steps
            dx2 = (act_clipped[1] - states[1, 0]) / self.time_delta_steps
            dx3 = (act_clipped[2] - states[2, 0]) / self.time_delta_steps
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0] + dx1, states[0, 1], 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0] + dx2, states[1, 1], 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[2, :3] = np.array([states[2, 0] + dx3, states[2, 1], 0.])
                states[2, 3:6] = np.array([states[2, 0], states[2, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))

            # move down to the ground
            for i in range(self.time_delta_steps):
                states[0, :3] = np.array([states[0, 0], states[0, 1] - self.cube1_dh, 0.])
                states[0, 3:6] = np.array([states[0, 0], states[0, 1], 0.])
                states[1, :3] = np.array([states[1, 0], states[1, 1] - self.cube2_dh, 0.])
                states[1, 3:6] = np.array([states[1, 0], states[1, 1], 0.])
                states[2, :3] = np.array([states[2, 0], states[2, 1] - self.cube3_dh, 0.])
                states[2, 3:6] = np.array([states[2, 0], states[2, 1], 0.])
                states[:, 6:10] = quat_curr
                pyflex.set_shape_states(states)
                pyflex.step(render=True)
                time.sleep(self.time_sleep)
                if self.recording:
                    self.video_frames.append(self.render(mode='rgb_array'))
            return

        quat_curr = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of cubes
        states = np.zeros((self.num_cubes, self.dim_shape_state))

        for i in range(self.num_cubes):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        # make action as increasement, clip its range (from PassWater environment)
        act_clipped = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        _update_current_states_with_prev_states(states)
        if self.num_picker == 1:
            act_clipped = act_clipped.item()
            if self.timestep % 3 == 0: # cube 1
                cube_index = 0
                cube_dh = self.cube1_dh
                self.cube1_x = act_clipped
            elif self.timestep % 3 == 1: # cube 2
                cube_index = 1
                cube_dh = self.cube2_dh
                self.cube2_x = act_clipped
            elif self.timestep % 3 == 2:# cube 3
                cube_index = 2
                cube_dh = self.cube3_dh
                self.cube3_x = act_clipped
            else:
                raise NotImplementedError
            _move_cube_with_animation(cube_index, cube_dh, states, act_clipped)

            # update final cube position
            states[cube_index, :3] = np.array([act_clipped, self.cube_y_coord, 0.])
        elif self.num_picker == 2:
            if self.timestep % 2 == 0: # cubes 1 and 2
                _move_two_cubes_with_animation(states, act_clipped)
                # update cubes position
                self.cube1_x = act_clipped[0]
                self.cube2_x = act_clipped[1]
            elif self.timestep % 2 == 1:
                _move_cube_with_animation(2, self.cube3_dh, states, np.array([act_clipped[0]]))
                # update cube position
                self.cube3_x = act_clipped[0]
        elif self.num_picker == 3:
            _move_three_cubes_with_animation(states, act_clipped)
            # update cubes position
            self.cube1_x = act_clipped[0]
            self.cube2_x = act_clipped[1]
            self.cube3_x = act_clipped[2]

        # for all cubes
        states[:, 6:10] = quat_curr
        self.glass_states = states

    def move_cube(self, prev_states, action):
        '''
        given the previous states of the glass, move it in 1D along x-axis.
        update the states of the 5 boxes that form the box: floor, left/right wall, back/front wall. 

        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat 
        '''
        quat_curr = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of cubes
        states = np.zeros((self.num_cubes, self.dim_shape_state))

        for i in range(self.num_cubes):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]
            # make sure we start the cubes on the ground rather than under it
            states[i][1] = self.cube_y_coord
            states[i][4] = self.cube_y_coord

        # make action as increasement, clip its range (from PassWater environment)
        act_clipped = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # print('act_clipped: ', act_clipped) # for debugging

        if self.num_picker == 1:
            act_clipped = act_clipped.item()
            if self.timestep % 3 == 0: # cube 1
                cube_index = 0
                self.cube1_x = act_clipped

                # make sure the other two cubes (cube2, cube3) keep the same values from the last timestep
                states[1][:3] = states[1][3:6]
                states[1][6:10] = states[1][10:]
                states[2][:3] = states[2][3:6]
                states[2][6:10] = states[2][10:]
            elif self.timestep % 3 == 1: # cube 2
                cube_index = 1
                self.cube2_x = act_clipped

                # make sure the other two cubes (cube1, cube3) keep the same values from the last timestep
                states[0][:3] = states[0][3:6]
                states[0][6:10] = states[0][10:]
                states[2][:3] = states[2][3:6]
                states[2][6:10] = states[2][10:]
            elif self.timestep % 3 == 2: # cube 3
                cube_index = 2
                self.cube3_x = act_clipped

                # make sure the other two cubes (cube1, cube2) keep the same values from the last timestep
                states[0][:3] = states[0][3:6]
                states[0][6:10] = states[0][10:]
                states[1][:3] = states[1][3:6]
                states[1][6:10] = states[1][10:]
            else:
                raise NotImplementedError

            # update cube position
            states[cube_index, :3] = np.array([act_clipped, self.cube_y_coord, 0.])
        elif self.num_picker == 2:
            if self.timestep % 2 == 0: # cubes 1 and 2
                self.cube1_x = act_clipped[0]
                self.cube2_x = act_clipped[1]
                # update cubes position
                states[0, :3] = np.array([self.cube1_x, self.cube_y_coord, 0.])
                states[1, :3] = np.array([self.cube2_x, self.cube_y_coord, 0.])

                # make sure cube3 keeps the same values from the last timestep
                states[2][:3] = states[2][3:6]
                states[2][6:10] = states[2][10:]
            elif self.timestep % 2 == 1: # cube 3
                self.cube3_x = act_clipped[0]
                # update cube3 position
                states[2, :3] = np.array([self.cube3_x, self.cube_y_coord, 0.])

                # make sure the other two cubes (cube1, cube2) keep the same values from the last timestep
                states[0][:3] = states[0][3:6]
                states[0][6:10] = states[0][10:]
                states[1][:3] = states[1][3:6]
                states[1][6:10] = states[1][10:]
        elif self.num_picker == 3:
            self.cube1_x = act_clipped[0]
            self.cube2_x = act_clipped[1]
            self.cube3_x = act_clipped[2]

            # update cubes position
            states[0, :3] = np.array([self.cube1_x, self.cube_y_coord, 0.])
            states[1, :3] = np.array([self.cube2_x, self.cube_y_coord, 0.])
            states[2, :3] = np.array([self.cube3_x, self.cube_y_coord, 0.])

        # for all cubes
        states[:, 6:10] = quat_curr

        self.glass_states = states

    def init_glass_state(self):
        '''
        set the initial state of the glass.
        '''
        y_curr = y_last = self.cube_y_coord
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of cubes
        states = np.zeros((self.num_cubes, self.dim_shape_state))

        # cube 1
        states[0, :3] = np.array([self.cube1_x, y_curr, 0.])
        states[0, 3:6] = np.array([self.cube1_x, y_last, 0.])

        # cube 2
        states[1, :3] = np.array([self.cube2_x, y_curr, 0.])
        states[1, 3:6] = np.array([self.cube2_x, y_last, 0.])

        # cube 3
        states[2, :3] = np.array([self.cube3_x, y_curr, 0.])
        states[2, 3:6] = np.array([self.cube3_x, y_last, 0.])

        # for all cubes
        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states

    def compute_random_actions(self):
        action = self.action_space.sample()
        action = self.normalize_action(action)
        return action

    def compute_expert_action(self):
        if self.num_picker == 3:
            action = np.array([1.1, 0.7, 0.1])
            action = self.normalize_action(action)
        else:
            raise NotImplementedError
        return action

    def normalize_action(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        act_normalized = (action - lb) / (ub - lb) * 2 - 1 # normalize to range [-1, 1]
        return act_normalized

if __name__ == '__main__':
    env = ThreeCubesEnv(observation_mode='cam_rgb',
                         action_mode='direct',
                         render=True,
                         headless=False,
                         horizon=75,
                         action_repeat=8,
                         render_mode='fluid',
                         deterministic=True)
    env.reset()
    for i in range(500):
        pyflex.step()
