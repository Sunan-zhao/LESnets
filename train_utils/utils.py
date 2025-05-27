import os
import numpy as np
import torch



def get_grid4d(S1, S2, S3, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S1 + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S1, 1, 1, 1, 1).repeat([1, 1, S2, S3, 1, T])
    gridy = torch.tensor(np.linspace(0, 1, S2 + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S2, 1, 1, 1).repeat([1, S1, 1, S3, 1, T])
    gridz = torch.tensor(np.linspace(0, 1, S3 + 1)[:-1], dtype=torch.float, device=device)
    gridz = gridz.reshape(1, 1, 1, S3, 1, 1).repeat([1, S1, S2, 1, 1, T])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, 1, 1, T).repeat([1, S1, S2, S3, 1, 1])
    return gridx, gridy, gridz, gridt



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def set_grad(tensors, flag=True):
    for p in tensors:
        p.requires_grad = flag


def zero_grad(params):
    '''
    set grad field to 0
    '''
    if isinstance(params, torch.Tensor):
        if params.grad is not None:
            params.grad.zero_()
    else:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()


def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)



def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None
    
    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save({
        'model': model_state, 
        'optim': optim_state, 
        'scheduler': scheduler_state
    }, path)
    print(f'Checkpoint is saved to {path}')


def dict2str(log_dict):
    res = ''
    for key, value in log_dict.items():
        res += f'{key}: {value}|'
    return res