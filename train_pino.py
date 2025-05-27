
import os
import yaml
import random
from argparse import ArgumentParser
from timeit import default_timer

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import FNO3d, IFNO3d
from train_utils.datasets import NS_3D_Dataset, sample_data
from train_utils.utils import save_ckpt, count_params, dict2str,get_grid4d
from train_utils.train_3d import train, eval_ns


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # flow type
    print("#" * 60)
    flow_name =  config['data']['name']
    LES_model_name = config['LES_model']['model_name']
    model_name = config['model']['name']
    loss_type = config['train']['loss_function_type']
    print('Flow Type:', flow_name)
    print('SGS Model:', LES_model_name)
    print('Neural Operator:',model_name)
    print('Loss Function Type:', loss_type)
    print("#" * 60)
    # model list
    model_name_list = ['FNO','IFNO']
    # create model
    if model_name == model_name_list[0]:
        model = FNO3d(layers=config['model']['layers'],
                      width = config['model']['Width'],
                      modes1=config['model']['modes1'],
                      modes2=config['model']['modes2'],
                      modes3=config['model']['modes3'],
                      modes4=config['model']['modes4'],
                      fc_dim=config['model']['fc_dim'],
                      act=config['model']['act'],
                      LES_coe = config['LES_model']['isKnown']).to(device)
    if model_name == model_name_list[1]:
        model = IFNO3d(layers=config['model']['layers'],
                      width = config['model']['Width'],
                      modes1=config['model']['modes1'],
                      modes2=config['model']['modes2'],
                      modes3=config['model']['modes3'],
                      modes4=config['model']['modes4'],
                      fc_dim=config['model']['fc_dim'],
                      act=config['model']['act'],
                      LES_coe = config['LES_model']['isKnown']).to(device)
    num_params = count_params(model)
    print(f'Number of parameters: {num_params}')
    total_memory = torch.cuda.get_device_properties(device).total_memory
    print(f"Total GPU Memory: {total_memory / 1024 ** 2:.2f} MB")

# Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print('Weights loaded from %s' % ckpt_path)
# For inference
    if args.predict:
        print("Predicting Started!")
        predict_data_path = config['predict']['paths']
        advance_steps = config['predict']['advance_steps']
        advamce_time = config['predict']['advance_time']
        predict_data = np.load(predict_data_path[0])
        predict_data = torch.from_numpy(predict_data)
        casenumber,T, X, Y, Z, dim = predict_data.size()
        predict_vel_t_total = torch.zeros([1, X, Y, Z, dim, advance_steps*(advamce_time)+1])

        for sample_id in range(casenumber):
            input_vel = predict_data[sample_id,...].permute(1,2,3,4,0).unsqueeze(0)
            predict_vel_t = input_vel
            predict_vel_t.cpu()
            gridx, gridy, gridz, gridt = get_grid4d(X,Y,Z,advamce_time)
            grid = torch.cat((gridx, gridy, gridz, gridt), dim=-2)
            input_vel = torch.cat((grid,input_vel.repeat(1,1,1,1,1,advamce_time)),dim = -2)
            t1 = default_timer()
            for i in range(advance_steps):
                print(i+1)
                predict_vel = model(input_vel.to(device)).detach().cpu()
                inpu_vel_new = predict_vel[...,-1].unsqueeze(-1)
                inpu_vel_new.cpu()
                predict_vel_t = torch.cat((predict_vel_t,predict_vel),dim = -1)
                gridx, gridy, gridz, gridt = get_grid4d(X,Y,Z,advamce_time)
                grid = torch.cat((gridx, gridy, gridz, gridt), dim=-2)
                input_vel = torch.cat((grid, inpu_vel_new.repeat(1, 1, 1, 1, 1, advamce_time)), dim=-2)
            predict_vel_t = predict_vel_t[:,:,:,:,:,:]
            predict_vel_t_total = torch.cat((predict_vel_t_total,predict_vel_t),dim = 0)
            t2 = default_timer()
            print('Predict Time',t2-t1)
        predict_vel_t_total = predict_vel_t_total[1:,...]
        predict_vel_t_total = predict_vel_t_total.permute(0,5,1,2,3,4).detach().cpu()
        print(predict_vel_t_total.size())
        datasize = config['data']['n_data_samples']
        layers = config['model']['layers']
        width = config['model']['Width']
        modes = config['model']['modes1']
        xy_weight = config['train']['xy_weight']
        cs_weight = config['train']['cs_weight']
        logdir = "L{}_W{}_M{}_data{}_g{}_gp{}".format(layers, width, modes, datasize, xy_weight, cs_weight)
        base_dir_1 = os.path.join(model_name, logdir)
        base_dir_2 = os.path.join(flow_name, base_dir_1)
        ckpt_dir = os.path.join('predict', base_dir_2)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'predict.npy')
        np.save(ckpt_path,predict_vel_t_total)
# For training
    else:
        batchsize = config['train']['batchsize']
        # training set
        u_set = NS_3D_Dataset(paths=config['data']['paths'],
                              dt=config['data']['dt'],
                              data_res=config['data']['data_res'],
                              pde_res=config['data']['pde_res'],
                              n_samples=config['data']['n_data_samples'],
                              offset=config['data']['offset'],
                              t_duration=config['data']['t_duration'])
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=4, shuffle=True)
        # val set
        val_set = NS_3D_Dataset(paths=config['test']['paths'],
                           dt=config['data']['dt'],
                            data_res=config['test']['data_res'],
                            pde_res=config['test']['pde_res'],
                            n_samples=config['test']['n_test_samples'],
                            offset=config['test']['testoffset'],
                            t_duration=config['data']['t_duration'])
        val_loader = DataLoader(val_set, batch_size=batchsize, num_workers=4)

        print(f'Train set: {len(u_set)}; Test set: {len(val_set)}')
        print("#" * 60)
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])

        train(model,batchsize,
                 u_loader,
                 val_loader,
                 optimizer, scheduler,
                 device,
                 config)

    print('Done!')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--predict', action='store_true', help='Predict')
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)