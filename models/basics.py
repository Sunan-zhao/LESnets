
import torch
import torch.nn as nn

@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyzt,ioxyzt->boxyzt", a, b)
    return res

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights5 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights6 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights7 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights8 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[ -4, -3, -2,-1])
        t_dim = self.modes4
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], x_ft.shape[4], t_dim,
                             device=x.device, dtype=torch.cfloat)
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding
        # +++
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :] = compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3,:t_dim], self.weights1)
        # -++
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :] = compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3,:t_dim], self.weights2)
        # +-+
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :] = compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3,:t_dim], self.weights3)
        # --+
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :] = compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3,:t_dim], self.weights4)
        # -+-
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :] = compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:,:t_dim], self.weights5)
        # ++-
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :] = compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :t_dim], self.weights6)
        # +--
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :] = compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :t_dim], self.weights7)
        # ---
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :] = compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:,:t_dim], self.weights8)
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4), x.size(5)), dim=[2, 3, 4, 5])
        return x



