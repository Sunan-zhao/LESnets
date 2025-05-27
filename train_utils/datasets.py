
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_grid4d

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
'''
dataset class for loading initial and following time series velocity flow fileds
'''
class NS_3D_Dataset(Dataset):
    def __init__(self, paths, dt,
                 data_res, pde_res,
                 n_samples=None,
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.data_res = data_res
        self.pde_res = pde_res
        self.dt = dt
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        self.load()


    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath)
        # initial data
        a_data = raw_data[self.offset: self.offset + self.n_samples, 0, ::, ::, ::, :]
        a_data = a_data.reshape(self.n_samples, 1, self.pde_res[0], self.pde_res[1], self.pde_res[2], 3)
        # subsample ratio
        K = self.t_duration = self.t_duration/self.dt
        if K == self.pde_res[3]:
            data = raw_data[self.offset: self.offset + self.n_samples, 1:, ...]
        else:
            sub_t = int(K/self.pde_res[3])
            data = raw_data[self.offset: self.offset + self.n_samples, 1::sub_t, ...]

        # convert into torch tensor
        data_copy = np.copy(data)
        data = torch.from_numpy(data_copy).to(torch.float32)
        a_data = torch.from_numpy(a_data).to(torch.float32).permute(0, 2, 3, 4, 5, 1)
        self.data = data.permute(0, 2, 3, 4, 5, 1)
        S = self.pde_res[1]

        gridx, gridy, gridz, gridt = get_grid4d(self.pde_res[0], self.pde_res[1], self.pde_res[2], self.pde_res[3])
        self.grid = torch.cat((gridx[0], gridy[0], gridz[0], gridt[0]), dim=-2)

        self.a_data = a_data

    def __getitem__(self, idx):
        a_data = torch.cat((self.grid,
            self.a_data[idx].repeat(1, 1, 1, 1, self.pde_res[3])), dim=-2)

        return a_data,self.data[idx]

    def __len__(self, ):
        return self.data.shape[0]
