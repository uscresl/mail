import numpy as np
from gym.spaces import Box
from gym import spaces
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import Picker, PickerPickAndPlace
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy


class RopeNewEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=75, render_mode='particle', picker_radius=0.03, picker_threshold=0.005, particle_radius=0.00625, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point', 'cam_rgb_key_point']
        assert action_mode in ['picker', 'pickerpickandplace', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_picker = num_picker
        self.enable_stack_frames = kwargs.get('enable_stack_frames', False)
        self.env_image_size = kwargs.get('env_image_size', 32)

        if observation_mode == 'cam_rgb' and self.enable_stack_frames:
            raise NotImplementedError

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, picker_threshold=picker_threshold, 
            particle_radius=0.025, picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)
        elif action_mode == 'pickerpickandplace':
            self.action_tool = PickerPickAndPlace(num_picker=num_picker, particle_radius=particle_radius, picker_radius=picker_radius, env=self,
                                               picker_low=(-0.35, 0., -0.35), picker_high=(0.35, 0.3, 0.35))
            self.action_space = self.action_tool.action_space
            assert self.action_repeat == 1

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 10 * 3
            else:
                max_particles = 41
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            if self.enable_stack_frames:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3, obs_dim), dtype=np.float32)
            else:
                self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)
        elif observation_mode == 'cam_rgb_key_point':
            obs_dim = 10 * 3
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = spaces.Dict(
                dict(
                    image = Box(low=-np.inf, high=np.inf, shape=(self.env_image_size, self.env_image_size, 3), dtype=np.float32),
                    key_point = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
                )
            )

        self.horizon = horizon
        # print("init of rope new env done!")

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'init_pos': [0., 0., 0.],
            'stretchstiffness': 0.9,
            'bendingstiffness': 0.8,
            'radius': 0.025,
            'segment': 40,
            'mass': 0.5,
            'scale': 0.5,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 0.85, 0]),
                                   'angle': np.array([0 * np.pi, -90 / 180. * np.pi, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb_key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            pos = keypoint_pos.flatten()
            if self.action_mode in ['sphere', 'picker']:
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
            pos = {
                'image': self.get_image(self.env_image_size, self.env_image_size),
                'key_point': pos,
            }
        else:
            if self.observation_mode == 'cam_rgb':
                return self.get_image(self.camera_height, self.camera_width)
            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
            elif self.observation_mode == 'key_point':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
                keypoint_pos = particle_pos[self.key_point_indices, :3]
                pos = keypoint_pos.flatten()
                # more_info = np.array([self.rope_length, self._get_endpoint_distance()])
                # pos = np.hstack([more_info, pos])
                # print("in _get_obs, pos is: ", pos)

            if self.action_mode in ['sphere', 'picker']:
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def _get_key_point_idx(self, num=None):
        indices = [0]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)
        indices.append(num - 1)

        return indices

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        camera_params = config['camera_params'][config['camera_name']]
        params = np.array(
            [*config['init_pos'], config['stretchstiffness'], config['bendingstiffness'], config['radius'], config['segment'], config['mass'], 
                config['scale'], 
                *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], render_mode]
            )

        env_idx = 2

        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (params, robot_params)
            pyflex.set_scene(env_idx, params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, params, 0)

        num_particles = pyflex.get_n_particles()
        # print("with {} segments, the number of particles are {}".format(config['segment'], num_particles))
        # exit()
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def _get_info(self):
        return {}

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def normalize_action(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        act_normalized = (action - lb) / (ub - lb) * 2 - 1 # normalize to range [-1, 1]
        return act_normalized

    def unnormalize_action(self, act_normalized):
        lb, ub = self.action_space.low, self.action_space.high
        act_unnormalized = ((act_normalized + 1) * (ub - lb)) / 2 + lb
        return act_unnormalized

    def normalize_action_pick_place(self, pick, place=np.array([0., 0., 0.])):
        assert pick.shape == (3,) # x,y,z location of pick
        scaled_action = np.concatenate([pick, place])
        lb, ub = self.action_space.low, self.action_space.high
        act_normalized = (scaled_action - lb) / (ub - lb) * 2 - 1 # normalize to range [-1, 1]
        return act_normalized


if __name__ == '__main__':
    env = RopeNewEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=10,
                  use_cached_states=False,
                  save_cached_states=False,
                  deterministic=False)
    env.reset(config=env.get_default_config())
    for i in range(1000):
        print(i)
        print("right before pyflex step")
        pyflex.step()
        print("right after pyflex step")
        print("right before pyflex render")
        pyflex.render()
        print("right after pyflex render")
