import torch.nn.functional as F
import numpy as np
import torch

def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func

def get_grid_4d(batchsize,S, device='cpu'): # S=32,T=11,time_scale=2.0
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([batchsize, 1, S, S, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([batchsize, S, 1, S, 1])
    gridz = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridz = gridz.reshape(1, 1, 1, S, 1).repeat([batchsize, S, S, 1, 1])
    # gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    # gridt = gridt.reshape(1, 1, 1, 1, T, 1).repeat([1, S, S, S, dim, 1])
    # gridw = gridw.tensor(np.linspace(0, 1, dim+1)[:-1],dtype=torch.float, device=device)
    # gridw = gridw.reshape(1, 1, 1, 1, dim, 1).repeat([1, S, S, S, 1, 1])

    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
