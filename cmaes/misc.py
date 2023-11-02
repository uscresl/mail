# import ipdb
import cv2
from enum import IntEnum
import json
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
# from mujoco_py import functions

class VizOptions(IntEnum):
  NONE = 0
  INTERACTIVE = 1
  VIDEO = 2

sqnorm = lambda x: np.inner(x, x) # squared 2-norm

# Decorator for static variables in python functions
def static_vars(**kwargs):
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate

# def quat_dist(q1: np.ndarray, q2: np.ndarray) -> float:
#   """Computes the distance between two unit quaternions."""
#   assert np.isclose(np.linalg.norm(q1), 1.)
#   assert np.isclose(np.linalg.norm(q2), 1.)
#   assert q1.size == 4
#   assert q2.size == 4

#   q2_inv = np.zeros(4)
#   q_diff = np.zeros(4)
#   functions.mju_negQuat(q2_inv, q2)
#   functions.mju_mulQuat(q_diff, q1, q2_inv)
#   if np.isclose(np.fabs(q_diff[0]), 1.):
#     return 0.
#   else:
    # return 2 * np.arctan2(np.linalg.norm(q_diff[1:]), q_diff[0])


def quat2rpy(q: np.ndarray): # quat [w x y z]
  """Convert quaternion to roll pitch and yaw. Following psuedocode from wikipedia
  https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
  """
  assert np.isclose(np.linalg.norm(q), 1.)
  assert q.size == 4

  # roll (x-axis rotation)
  sinr_cosp = 2*(q[0]*q[1] + q[2]*q[3])
  cosr_cosp = 1 - 2*(q[1]**2 + q[2]**2)
  roll = np.math.atan2(sinr_cosp, cosr_cosp)

  # pitch (y-axis rotation)
  sinp = 2 * (q[0]*q[2] - q[3]*q[1])
  pitch = np.math.asin(sinp)

  # yaw (z-axis rotation)
  siny_cosp = 2*(q[0]*q[3] + q[1]*q[2])
  cosy_cosp = 1 - 2*(q[2]**2 + q[3]**2)
  yaw = np.math.atan2(siny_cosp, cosy_cosp)

  return (roll, pitch, yaw)


def save_video(fpath, frames, fps=8.):
  import moviepy.editor as mpy
  def get_frame(t):
      frame_length = len(frames)
      new_fps = 1./(1./fps + 1./frame_length)
      idx = min(int(t*new_fps), frame_length-1)
      return frames[idx]

  video = mpy.VideoClip(get_frame, duration=len(frames)/fps+2)
  video.write_videofile(fpath, fps, logger=None) # 'verbose' arg is deprecated
  print(f"[*] Video saved: {fpath}. Num frames: {len(frames)}")

def change_mjcf_softness(xml_path: str, new_k: float, new_d: float, prefix:str="model"):
  """Change mjcf softness params (solref, solimp)
  assume mjcf is organized as follows
  <mujoco>
    <default ...> # to set all defaults
      <geom solref="..." solimp="..." ... >
    </default>
  ... # rest of model
  </mujoco>
  """
  # parse model
  tree = ET.parse(xml_path)
  root = tree.getroot()
  assert(root.tag == 'mujoco') #Check if mujoco model

  # set solref params
  elem_default_geom = root.find('default').find('geom')
  elem_default_geom.set('solref', f'-{new_k} -{new_d}')
  elem_default_geom.set('solimp', '0.0 1.0 0.01')

  # write to new xml
  xml_dir = os.path.dirname(xml_path)
  outfile = f'{prefix}_k{new_k}_d{new_d}.xml'
  outpath = os.path.join(xml_dir, outfile)
  tree.write(outpath)
  print(f"Wrote new xml to {outpath}")
  return outpath


def change_body_height(xml_path: str, body_name:str, new_ht: float, prefix:str="model"):
  """
  Change z position of body 'body_name' to 'new_ht'
  Example usage:
  orig_xml = /path/to/halfcheetahsoft.xml
  new_xml = change_body_height(orig_xml, 'step0', 0.4, prefix='half_cheetah')
  env = gym.make("half-cheetah-soft-v0", xml_file=new_xml) # or define env some other way, maybe with a wrapper
  """
  xml_dir = os.path.dirname(xml_path)
  outfile = f'{prefix}_{body_name}_ht_{new_ht}.xml'
  outpath = os.path.join(xml_dir, outfile)
  if not os.path.isfile(outpath):
    # parse model
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert(root.tag == 'mujoco') #Check if mujoco model
    assert(isinstance(body_name, str))

    # set body height
    bodies_iter = root.iter('body') # iterator for all bodies
    for body in bodies_iter:
      if body.get('name') == body_name:
        pos_list = body.get('pos').split(" ")
        body.set('pos', f'{pos_list[0]} {pos_list[1]} {new_ht}')
        break

    # write to new xml
    tree.write(outpath)
    print(f"Wrote new xml to {outpath}")
  return outpath


def trajs_to_videos(npz_path:str, prob, save_dir:str = "."):
  trajs = np.load(npz_path)
  state_trajs = trajs['states']
  i:int = 0
  for state_traj in state_trajs:
    prob.save_video(state_traj, debug = True, vid_path=f"{save_dir}/{i}_states.mp4")
    i = i+1
  print(f"Saved {i} videos")

class NumpyArrayEncoder(json.JSONEncoder):
  """ Encode numpy arrays as lists. Use with 'cls' kwarg:
  numpyData = {"array": my_numpy_array}
  encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder) # dump to string
  json.dump(numpyData, file, cls=NumpyArrayEncoder) # dump to file
  """
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, np.int64):
      return int(obj)
    return json.JSONEncoder.default(self, obj)

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
}