import torch

"""
Data in goal_prox_il/expert_datasets/nav_100.pt:


          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]],

         [[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]]), 'done': tensor([0., 0., 0.,  ..., 0., 0., 1.]), 'actions': tensor([[1],
        [1],
        [1],
        ...,
        [3],
        [3],
        [2]]), 'ep_found_goal': tensor([0., 0., 0.,  ..., 0., 0., 1.])}
(Pdb) p data.keys()
dict_keys(['obs', 'next_obs', 'done', 'actions', 'ep_found_goal'])
(Pdb) p data['obs'].shape
torch.Size([13619, 4, 19, 19])
(Pdb) p data['done'].shape
torch.Size([13619])
(Pdb) p data['actions'].shape
torch.Size([13619, 1])
"""

data = torch.load('goal_prox_il/expert_datasets/nav_100.pt')

breakpoint()
print()

