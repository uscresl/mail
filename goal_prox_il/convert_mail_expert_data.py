import torch
import numpy as np
import argparse
from sb3.utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--mail_data_filepath', default='debug.pkl', type=str, help="mail_data_filepath")
parser.add_argument('--out_filename', default='debug.pt', type=str, help="out_filename")
parser.add_argument('--should_get_first_hundred', default=False, type=str2bool, help="only get the first one hundred episodes of data")
parser.add_argument('--is_state_based', default=False, type=str2bool, help="is state-based observation")

args = parser.parse_args()

mail_data_filepath = args.mail_data_filepath
out_filename = args.out_filename

mail_data = np.load(mail_data_filepath, allow_pickle=True)

if args.should_get_first_hundred:
    mail_data['done_trajs'] = mail_data['done_trajs'][:100]
    if args.is_state_based:
        mail_data['ob_trajs'] = mail_data['ob_trajs'][:100]
        mail_data['ob_next_trajs'] = mail_data['ob_next_trajs'][:100]
    else:
        mail_data['ob_img_trajs'] = mail_data['ob_img_trajs'][:100]
        mail_data['ob_img_next_trajs'] = mail_data['ob_img_next_trajs'][:100]
    mail_data['action_trajs'] = mail_data['action_trajs'][:100]

# construct ep_found_goal data
ep_found_goal = []
for traj in mail_data['done_trajs']:
    traj_data = []
    for true_false in traj:
        if true_false:
            traj_data.append(float(1.0))
        else:
            traj_data.append(float(0.0))
    ep_found_goal.append(traj_data)
ep_found_goal = np.array(ep_found_goal)

if args.is_state_based:
    mail_data['ob_trajs'] = np.reshape(mail_data['ob_trajs'], (int(mail_data['ob_trajs'].shape[0] * mail_data['ob_trajs'].shape[1]), mail_data['ob_trajs'].shape[2]))
    mail_data['ob_next_trajs'] = np.reshape(mail_data['ob_next_trajs'], (int(mail_data['ob_next_trajs'].shape[0] * mail_data['ob_next_trajs'].shape[1]), mail_data['ob_next_trajs'].shape[2]))
else:
    # reshape from (100, 3, 32, 32, 3) to (300, 32, 32, 3)
    mail_data['ob_img_trajs'] = np.reshape(mail_data['ob_img_trajs'], (int(mail_data['ob_img_trajs'].shape[0] * mail_data['ob_img_trajs'].shape[1]), mail_data['ob_img_trajs'].shape[2], mail_data['ob_img_trajs'].shape[3], mail_data['ob_img_trajs'].shape[4]))
    mail_data['ob_img_next_trajs'] = np.reshape(mail_data['ob_img_next_trajs'], (int(mail_data['ob_img_next_trajs'].shape[0] * mail_data['ob_img_next_trajs'].shape[1]), mail_data['ob_img_next_trajs'].shape[2], mail_data['ob_img_next_trajs'].shape[3], mail_data['ob_img_next_trajs'].shape[4]))

    # transpose from (300, 32, 32, 3) to (300, 3, 32, 32)
    mail_data['ob_img_trajs'] = np.transpose(mail_data['ob_img_trajs'], (0, 3, 1, 2))
    mail_data['ob_img_next_trajs'] = np.transpose(mail_data['ob_img_next_trajs'], (0, 3, 1, 2))

mail_data['done_trajs'] = np.reshape(mail_data['done_trajs'], (int(mail_data['done_trajs'].shape[0] * mail_data['done_trajs'].shape[1])))
mail_data['action_trajs'] = np.reshape(mail_data['action_trajs'], (int(mail_data['action_trajs'].shape[0] * mail_data['action_trajs'].shape[1]), mail_data['action_trajs'].shape[2]))
ep_found_goal = np.reshape(ep_found_goal, (int(ep_found_goal.shape[0] * ep_found_goal.shape[1])))

# construct goal_prox_data
goal_prox_data = {}
if args.is_state_based:
    goal_prox_data['obs'] = torch.from_numpy(mail_data['ob_trajs'])
    goal_prox_data['next_obs'] = torch.from_numpy(mail_data['ob_next_trajs'])
else:
    goal_prox_data['obs'] = torch.from_numpy(mail_data['ob_img_trajs'])
    goal_prox_data['next_obs'] = torch.from_numpy(mail_data['ob_img_next_trajs'])
goal_prox_data['done'] = torch.from_numpy(mail_data['done_trajs'])
goal_prox_data['actions'] = torch.from_numpy(mail_data['action_trajs'])
goal_prox_data['ep_found_goal'] = torch.from_numpy(ep_found_goal)

torch.save(goal_prox_data, out_filename)
print(f'{out_filename} saved!!')
