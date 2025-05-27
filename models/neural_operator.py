import torch
import torch.nn as nn
from .basics import SpectralConv3d
from .utils import _get_act


class FNO3d(nn.Module):
    '''
    Args:
        layers: int,
            numbers of Fourier layer
        Width: int,
            channel space for each Fourier layer
        modes1: int,
            first dimension maximal modes for each layer
        modes2: int,
            second dimension maximal modes for each layer
        modes3: int,
            third dimension maximal modes for each layer
        modes4: int,
            four dimension maximal modes for each layer
        fc_dim: int,
            dimension of fully connected layers
        in_dim: int,
            input dimension (three velocity components: 3 + grid information: 4)
        out_dim: int,
            output dimension (back to three velocity components: 3)
        act: str,
            {tanh, gelu, relu, leaky_relu}, activation function
        LES_coe: bool,
            whether the coefficent of LES model is known
    '''
    def __init__(
            self,
            layers: int,
            width: int=64,
            modes1: int=12,
            modes2: int=12,
            modes3: int=12,
            modes4: int=1,
            fc_dim: int=128,
            in_dim: int=7,
            out_dim: int=3,
            act: str='gelu',
            LES_coe: bool=True):
        super(FNO3d, self).__init__()

        self.modes1 = [modes1]*layers
        self.modes2 = [modes2]*layers
        self.modes3 = [modes3]*layers
        self.modes4 = [modes4]*layers
        self.layers = [width]*(layers+1)
        self.LES_coe = LES_coe
        self.fc0 = nn.Linear(in_dim, width)
        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3, self.modes4)])
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)
        if LES_coe is False:
            self.param = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):
        '''
        input:
            u1,u1,...,u1: (batchsize, x_grid, y_grid, z_grid, 3+4, t_grid),
        output:
            u1,u2,...,uT: (batchsize, x_grid, y_grid, z_grid, 3, t_grid).
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        x = x.permute(0, 1, 2, 3, 5, 4)
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        size_x, size_y, size_z,  size_t =  x.shape[-4],x.shape[-3], x.shape[-2], x.shape[-1]
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.reshape(batchsize, self.layers[i], -1)).reshape(batchsize, self.layers[i+1], size_x, size_y, size_z, size_t)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0,1,2,3,5,4)
        if self.LES_coe is True:
            return x
        else:
            Cs_square = self.param
            return x,Cs_square



class IFNO3d(nn.Module):
    '''
    Args:
        layers: int,
            numbers of Fourier layer (implicit)
        Width: int,
            channel space for each Fourier layer
        modes1: int,
            first dimension maximal modes for each layer
        modes2: int,
            second dimension maximal modes for each layer
        modes3: int,
            third dimension maximal modes for each layer
        modes4: int,
            four dimension maximal modes for each layer
        fc_dim: int,
            dimension of fully connected layers
        in_dim: int,
            input dimension (three velocity components: 3 + grid information: 4)
        out_dim: int,
            output dimension (back to three velocity components: 3)
        act: str,
            {tanh, gelu, relu, leaky_relu}, activation function
        LES_coe: bool,
            whether the coefficent of LES model is known
    '''
    def __init__(
            self,
            layers: int,
            width: int=64,
            modes1: int=12,
            modes2: int=12,
            modes3: int=12,
            modes4: int=1,
            fc_dim: int=128,
            in_dim: int=7,
            out_dim: int=3,
            act: str='gelu',
            LES_coe: bool=True):
        super(IFNO3d, self).__init__()

        self.modes1 = [modes1]
        self.modes2 = [modes2]
        self.modes3 = [modes3]
        self.modes4 = [modes4]
        self.layers = [width]*(1+1)
        self.nlayer = layers
        self.LES_coe = LES_coe
        self.fc0 = nn.Linear(in_dim, width)
        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num, mode4_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3, self.modes4)])
        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)
        if LES_coe is False:
            self.param = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):
        '''
        input:
            u1,u1,...,u1: (batchsize, x_grid, y_grid, z_grid, 3+4, t_grid),
        output:
            u1,u2,...,uT: (batchsize, x_grid, y_grid, z_grid, 3, t_grid).
        '''
        coef = 1./self.nlayer
        length = len(self.ws)
        batchsize = x.shape[0]
        x = x.permute(0, 1, 2, 3, 5, 4)
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        size_x, size_y, size_z,  size_t =  x.shape[-4],x.shape[-3], x.shape[-2], x.shape[-1]
        # implicit learning
        for j in range(self.nlayer - 1):
            x1 = self.sp_convs[0](x)
            x2 = self.ws[0](x.reshape(batchsize, self.layers[0], -1)).reshape(batchsize, self.layers[0], size_x, size_y,
                                                                              size_z, size_t)
            x = self.act(x1 + x2) * coef + x
        # not using act for last layer
        # x1 = self.sp_convs[0](x)
        # x2 = self.ws[0](x.reshape(batchsize, self.layers[0], -1)).reshape(batchsize, self.layers[0], size_x, size_y,
        #                                                                   size_z, size_t)
        # x = (x1 + x2) * coef + x
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.reshape(batchsize, self.layers[i], -1)).reshape(batchsize, self.layers[i+1], size_x, size_y, size_z, size_t)

            x = (x1 + x2)*coef + x
            if i != length - 1:
                x = self.act(x)
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0,1,2,3,5,4)
        if self.LES_coe is True:
            return x
        else:
            Cs_square = self.param
            return x,Cs_square
