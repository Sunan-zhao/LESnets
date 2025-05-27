import os
import yaml
import torch
import numpy as np
from timeit import default_timer
from torch.optim import Adam
from .losses import LpLoss, PINO_loss3d
from .data_utils import sample_data
from .utils import save_ckpt, count_params, dict2str
try:
    import wandb
except ImportError:
    wandb = None
    
@torch.no_grad()
def eval_ns(model, val_loader, criterion, device):
    model.eval()
    val_err = []
    for a, u in val_loader:
        a, u = a.to(device), u.to(device)
        out = model(a)
        val_loss = criterion(out, u)
        val_err.append(val_loss.item())
    N = len(val_loader)
    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return avg_err, std_err

@torch.no_grad()
def eval_ns_cs(model, val_loader, criterion, device):
    model.eval()
    val_err = []
    for a, u in val_loader:
        a, u = a.to(device), u.to(device)
        out,_ = model(a)
        val_loss = criterion(out, u)
        val_err.append(val_loss.item())
    N = len(val_loader)
    avg_err = np.mean(val_err)
    std_err = np.std(val_err, ddof=1) / np.sqrt(N)
    return avg_err, std_err

def train(model,batchsize,
             train_u_loader,
             val_loader,
             optimizer,
             scheduler,
             device, config):
# Data parameters
    v = config['data']['nv']
    pde_res = config['data']['pde_res']
    t_duration = config['data']['t_duration']
# Model parameters
    loss_function_type = config['train']['loss_function_type']
    datasize = config['data']['n_data_samples']
    layers = config['model']['layers']
    width = config['model']['Width']
    modes = config['model']['modes1']
# Loss weight
    xy_weight = config['train']['xy_weight']
    cs_weight = config['train']['cs_weight']
# Iteration steps
    start_iter = config['train']['start_iter']
    num_iter =  config['train']['num_iter']
    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']
    satrt_step = config['train']['start_step']
    pbar = range(start_iter, num_iter)
# Set up basic directory
    flow_name = config['data']['name']
    model_name = config['model']['name']
    logdir = "L{}_W{}_M{}_data{}_g{}_gp{}".format(layers, width, modes, datasize,xy_weight,cs_weight)
    if  cs_weight !=0:
        added_data_type = config['data']['added_data_type']
        logdir =  "{}_{}".format(logdir,added_data_type)
    print("*" * 40)
    print(f"Datasize: {datasize}  Layers: {layers}  Width: {width}  Mode: {modes}")
    print(f"data loss weight: {xy_weight}  Cs loss weight: {cs_weight}")
    print("*" * 40)
# Loss type
    lploss = LpLoss(size_average=True)
# Sample data
    u_loader = sample_data(train_u_loader)
# For physics-informed methods
    if loss_function_type == 'PI':
        model_name = 'PI' + model_name
    # Coefficent is known
        if config['LES_model']['isKnown'] is True:
            Cs_square = config['LES_model']['SM']['Cs_square']
        # Set up directory
            Cs_str = 'Cs_known'
            print(f"Cs is known")
            print("*" * 40)
            logdir = os.path.join(Cs_str, logdir)
            base_dir_1 = os.path.join(model_name, logdir)
            base_dir_2 = os.path.join(flow_name, base_dir_1)
            base_dir = os.path.join('exp', base_dir_2)
            ckpt_dir = os.path.join(base_dir, 'ckpts')
            loss_dir = os.path.join(base_dir, 'loss')
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(loss_dir, exist_ok=True)
            test_loss_list = []
            PDE_loss_list = []
            print("=" * 20)
            print("Training Started")
            print("=" * 20)
        # Training process
            for e in pbar:
                t1 = default_timer()
                optimizer.zero_grad()
            # model in and out
                a,_ = next(u_loader)
                a = a.to(device)
                out = model(a)
            # new time series
                u0 = a[:, :, :, :, -3:, 0].reshape(batchsize, pde_res[0], pde_res[1], pde_res[2], 3, 1)
                out = torch.cat((u0, out), dim=-1)
            # pde loss
                PDE_loss = PINO_loss3d(out, v, Cs_square, t_duration,flow_name)
            # loss function
                loss = PDE_loss
            # back propagation
                loss.backward()
                optimizer.step()
                scheduler.step()

                PDE_loss_list.append(PDE_loss.item())
                t2 = default_timer()
            # basic step
                if e % eval_step != 0:
                    print(f"Epoch: {e:05d}/{num_iter-1}  time: {(t2 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}")
            # evaluate step
                if e % eval_step == 0:
                    eval_err, std_err = eval_ns(model, val_loader, lploss, device)
                    t3 = default_timer()
                    test_loss_list.append(eval_err.item())
                    print(f"Epoch: {e:05d}/{num_iter-1}  time: {(t3 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}  Test Loss: {eval_err:.6f}")
            # save step
                if e % save_step == 0 and e > satrt_step:
                    ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
                    save_ckpt(ckpt_path, model, optimizer, scheduler)
                    print(f"Saving model at Epoch {e}!")
            print("=" * 20)
            print("Training Completed")
            print("=" * 20)
            print(f"Model parameter and loss data are saved in " + repr(base_dir))
        # Saving .dat
            PDE_loss_path = os.path.join(loss_dir, f'PDE_loss.dat')
            test_loss_path = os.path.join(loss_dir, f'test_loss.dat')
            np.savetxt(PDE_loss_path, PDE_loss_list, fmt="%16.7f")
            np.savetxt(test_loss_path, test_loss_list, fmt="%16.7f")

        # Coefficent is unknown
        else:
            # two seperate optimizers
            all_parameters = list(model.parameters())
            params_to_optimize = [param for param in all_parameters if param is not model.param]
            optimizer = Adam(params_to_optimize, lr=config['train']['base_lr'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config['train']['milestones'],
                                                             gamma=config['train']['scheduler_gamma'])
            optimizer_cs = Adam([model.param], lr=config['train']['cs_lr'])
            scheduler_cs = torch.optim.lr_scheduler.MultiStepLR(optimizer_cs,
                                                                milestones=config['train']['milestones'],
                                                                gamma=config['train']['scheduler_gamma'])
        # Additional dataset
            paths = config['data']['paths_2']
            cs_data = np.load(paths[0])
            cs_data = torch.from_numpy(cs_data)
            cs_data = cs_data.to(device)
            cs_data = cs_data.permute(0, 2, 3, 4, 5, 1)
        # Set up directory
            Cs_str = 'Cs_unknown'
            print(f"Cs is unknown")
            print("*" * 40)
            cs_lr = config['train']['start_iter']
            logdir = '{}_cslr{}'.format(logdir,cs_lr)
            logdir = os.path.join(Cs_str, logdir)
            base_dir_1 = os.path.join(model_name, logdir)
            base_dir_2 = os.path.join(flow_name, base_dir_1)
            base_dir = os.path.join('exp', base_dir_2)
            ckpt_dir = os.path.join(base_dir, 'ckpts')
            loss_dir = os.path.join(base_dir, 'loss')
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(loss_dir, exist_ok=True)
            test_loss_list = []
            PDE_loss_list = []
            Cs_loss_list = []
            Cs_square_list = []
            print("=" * 20)
            print("Training Started")
            print("=" * 20)
        # Training process
            for e in pbar:
                t1 = default_timer()
                optimizer.zero_grad()
                optimizer_cs.zero_grad()
            # model in and out
                a,_ = next(u_loader)
                a = a.to(device)
                out,Cs_square = model(a)
            # new time series
                u0 = a[:, :, :, :, -3:, 0].reshape(batchsize, pde_res[0], pde_res[1], pde_res[2], 3, 1)
                out = torch.cat((u0, out), dim=-1)
            # pde loss and Cs loss
                PDE_loss = PINO_loss3d(out, v, Cs_square, t_duration,flow_name)
                Cs_loss = PINO_loss3d(cs_data, Cs_square, v, t_duration,flow_name)
            # loss function
                loss = PDE_loss + cs_weight * Cs_loss
            # back propagation
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer_cs.step()
                scheduler_cs.step()

                PDE_loss_list.append(PDE_loss.item())
                Cs_loss_list.append(Cs_loss.item())
                Cs_square_list.append(Cs_square.item())
                t2 = default_timer()
            # basic step
                if e % eval_step != 0:
                    print(f"Epoch: {e:05d}/{num_iter - 1}  time: {(t2 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}")
            # evaluate step
                if e % eval_step == 0:
                    eval_err, std_err = eval_ns_cs(model, val_loader, lploss, device)
                    t3 = default_timer()
                    test_loss_list.append(eval_err.item())
                    print(
                        f"Epoch: {e:05d}/{num_iter - 1}  time: {(t3 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}  Cs Loss: {Cs_loss:.6f}  Cs^2: {Cs_square:.6f}  Test Loss: {eval_err:.6f}")
            # save step
                if e % save_step == 0 and e > satrt_step:
                    ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
                    save_ckpt(ckpt_path, model, optimizer, scheduler)
                    print(f"Saving model at Epoch {e}!")
            print("=" * 20)
            print("Training Completed")
            print("=" * 20)
            print(f"Model parameter and loss data are saved in " + repr(base_dir))
        # Saving .dat
            PDE_loss_path = os.path.join(loss_dir, f'PDE_loss.dat')
            test_loss_path = os.path.join(loss_dir, f'test_loss.dat')
            Cs_loss_path = os.path.join(loss_dir, f'Cs_loss.dat')
            Cs_square_path = os.path.join(loss_dir, f'Cs_square.dat')

            np.savetxt(PDE_loss_path, PDE_loss_list, fmt="%16.7f")
            np.savetxt(test_loss_path, test_loss_list, fmt="%16.7f")
            np.savetxt(Cs_loss_path, Cs_loss_list, fmt="%16.7f")
            np.savetxt(Cs_square_path, Cs_square_list, fmt="%16.7f")
# For data driven method
    if loss_function_type == 'Data_driven':
        base_dir_1 = os.path.join(model_name, logdir)
        base_dir_2 = os.path.join(flow_name, base_dir_1)
        base_dir = os.path.join('exp', base_dir_2)
        ckpt_dir = os.path.join(base_dir, 'ckpts')
        loss_dir = os.path.join(base_dir, 'loss')
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        test_loss_list = []
        data_loss_list = []
        print("=" * 20)
        print("Training Started")
        print("=" * 20)
        for e in pbar:
            t1 = default_timer()
            optimizer.zero_grad()
        # model in and out
            a, u = next(u_loader)
            a, u = a.to(device), u.to(device)
            out = model(a)
        # data loss
            data_loss = lploss(out, u)
        # loss function
            loss = data_loss
        # back propagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            data_loss_list.append(data_loss.item())
            t2 = default_timer()
        # basic step
            if e % eval_step != 0:
                print(f"Epoch: {e:05d}/{num_iter - 1}  time: {(t2 - t1):.2f}s  Data Loss: {data_loss:.6f}")
        # evaluate step
            if e % eval_step == 0:
                eval_err, std_err = eval_ns(model, val_loader, lploss, device)
                t3 = default_timer()
                test_loss_list.append(eval_err.item())
                print(
                    f"Epoch: {e:05d}/{num_iter - 1}  time: {(t3 - t1):.2f}s  Data Loss: {data_loss:.6f}  Test Loss: {eval_err:.6f}")
        # save step
            if e % save_step == 0 and e > satrt_step:
                ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
                save_ckpt(ckpt_path, model, optimizer, scheduler)
                print(f"Saving model at Epoch {e}!")
        print("=" * 20)
        print("Training Completed")
        print("=" * 20)
        print(f"Model parameter and loss data are saved in " + repr(base_dir))
    # Saving .dat
        data_loss_path = os.path.join(loss_dir, f'PDE_loss.dat')
        test_loss_path = os.path.join(loss_dir, f'test_loss.dat')
        np.savetxt(data_loss_path, data_loss_list, fmt="%16.7f")
        np.savetxt(test_loss_path, test_loss_list, fmt="%16.7f")
# For combined PI and data driven method
    if loss_function_type == 'PI_and_Data_driven':
        Cs_square = config['LES_model']['SM']['Cs_square']
    # Set up directory
        Cs_str = 'Cs_known'
        print(f"Cs is known")
        print("*" * 40)
        logdir = os.path.join(Cs_str, logdir)
        base_dir_1 = os.path.join(model_name, logdir)
        base_dir_2 = os.path.join(flow_name, base_dir_1)
        base_dir = os.path.join('exp', base_dir_2)
        ckpt_dir = os.path.join(base_dir, 'ckpts')
        loss_dir = os.path.join(base_dir, 'loss')
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(loss_dir, exist_ok=True)
        test_loss_list = []
        PDE_loss_list = []
        data_loss_list = []
        print("=" * 20)
        print("Training Started")
        print("=" * 20)
        for e in pbar:
            t1 = default_timer()
            optimizer.zero_grad()
        # model in and out
            a, u = next(u_loader)
            a, u = a.to(device), u.to(device)
            out = model(a)
        # new time series
            u0 = a[:, :, :, :, -3:, 0].reshape(batchsize, pde_res[0], pde_res[1], pde_res[2], 3, 1)
            out_new = torch.cat((u0, out), dim=-1)
        # pde loss
            PDE_loss = PINO_loss3d(out_new, v, Cs_square, t_duration,flow_name)
        # data loss
            data_loss = lploss(out, u)
        # loss function
            loss = PDE_loss + xy_weight * data_loss
        # back propagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            PDE_loss_list.append(PDE_loss.item())
            data_loss_list.append(data_loss.item())
            t2 = default_timer()
        # basic step
            if e % eval_step != 0:
                print(f"Epoch: {e:05d}/{num_iter - 1}  time: {(t2 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}  Data Loss: {data_loss:.6f}")
        # evaluate step
            if e % eval_step == 0:
                eval_err, std_err = eval_ns(model, val_loader, lploss, device)
                t3 = default_timer()
                test_loss_list.append(eval_err.item())
                print(
                    f"Epoch: {e:05d}/{num_iter - 1}  time: {(t3 - t1):.2f}s  PDE Loss: {PDE_loss:.6f}  Data Loss: {data_loss:.6f}  Test Loss: {eval_err:.6f}")
        # save step
            if e % save_step == 0 and e > satrt_step:
                ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
                save_ckpt(ckpt_path, model, optimizer, scheduler)
                print(f"Saving model at Epoch {e}!")
        print("=" * 20)
        print("Training Completed")
        print("=" * 20)
        print(f"Model parameter and loss data are saved in " + repr(base_dir))
    # Saving .dat
        PDE_loss_path = os.path.join(loss_dir, f'PDE_loss.dat')
        data_loss_path = os.path.join(loss_dir, f'PDE_loss.dat')
        test_loss_path = os.path.join(loss_dir, f'test_loss.dat')
        np.savetxt(PDE_loss_path, PDE_loss_list, fmt="%16.7f")
        np.savetxt(data_loss_path, data_loss_list, fmt="%16.7f")
        np.savetxt(test_loss_path, test_loss_list, fmt="%16.7f")



