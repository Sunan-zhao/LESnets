import numpy as np
import torch
import torch.nn.functional as F


def DHIT_Spectral_SM3d(u,Cs_square=0.1, v=1/40, t_interval=1.0):
    batchsize, nx, ny, nz, nc, nt  = u.shape
    device = u.device
    k_max = nx // 2
    filter_size = (2*np.pi)/32
    # Velocity component
    ux = u[:, :, :, :, 0, :]
    uy = u[:, :, :, :, 1, :]
    uz = u[:, :, :, :, 2, :]
    ux = ux.reshape(batchsize, nx, ny, nz, 1, nt)
    uy = uy.reshape(batchsize, nx, ny, nz, 1, nt)
    uz = uz.reshape(batchsize, nx, ny, nz, 1, nt)
    ux_h = torch.fft.fftn(ux, dim=[1, 2, 3], norm='forward')
    uy_h = torch.fft.fftn(uy, dim=[1, 2, 3], norm='forward')
    uz_h = torch.fft.fftn(uz, dim=[1, 2, 3], norm='forward')
    # Define wave numbers in x, y, and z directions
    kx = torch.fft.fftfreq(nx, d=1 / nx).to(device)
    ky = torch.fft.fftfreq(ny, d=1 / ny).to(device)
    kz = torch.fft.fftfreq(nz, d=1 / nz).to(device)
    kx = kx.view(1,nx, 1, 1, 1, 1).repeat(batchsize,1, ny, nz, 1, nt)
    ky = ky.view(1,1, ny, 1, 1, 1).repeat(batchsize,nx, 1, nz, 1, nt)
    kz = kz.view(1,1, 1, nz, 1, 1).repeat(batchsize,nx, ny, 1, 1, nt)
    lap = kx ** 2 + ky ** 2 + kz ** 2
    lap[:,0, 0, 0, 0, :] = 1
    # Vorticity component
    w_x_h = 1j * (ky * uz_h - kz * uy_h)
    w_y_h = 1j * (kz * ux_h - kx * uz_h)
    w_z_h = 1j * (kx * uy_h - ky * ux_h)
    w_x = torch.fft.irfftn(w_x_h[:, :, :, :k_max + 1, :], dim=[1, 2, 3], norm='forward').real
    w_y = torch.fft.irfftn(w_y_h[:, :, :, :k_max + 1, :], dim=[1, 2, 3], norm='forward').real
    w_z = torch.fft.irfftn(w_z_h[:, :, :, :k_max + 1, :], dim=[1, 2, 3], norm='forward').real
    # Compute derivatives in Fourier space
    ux_dx_h = 1j * kx * ux_h
    uy_dy_h = 1j * ky * uy_h
    uz_dz_h = 1j * kz * uz_h
    # Strain rate
    s11_h = ux_dx_h
    s22_h = uy_dy_h
    s33_h = uz_dz_h
    s23_h = 0.5 * 1j * (kz * uy_h + ky * uz_h)
    s13_h = 0.5 * 1j * (kx * uz_h + kz * ux_h)
    s12_h = 0.5 * 1j * (ky * ux_h + kx * uy_h)
    s11 = torch.fft.ifftn(s11_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    s22 = torch.fft.ifftn(s22_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    s33 = torch.fft.ifftn(s33_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    s23 = torch.fft.ifftn(s23_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    s13 = torch.fft.ifftn(s13_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    s12 = torch.fft.ifftn(s12_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    # Characteristic strain rate
    SS = 2.0 * (torch.complex((s11.real) * (s11.real), (s11.imag) * (s11.imag))
                + torch.complex((s22.real) ** 2, (s22.imag) ** 2)
                + torch.complex((s33.real) ** 2, (s33.imag) ** 2)
                + 2.0 * (torch.complex((s23.real) ** 2, (s23.imag) ** 2)
                         + torch.complex((s13.real) ** 2, (s13.imag) ** 2)
                         + torch.complex((s12.real) ** 2, (s12.imag) ** 2)))
    SS = torch.complex(torch.sqrt(SS.real), torch.sqrt(SS.imag))

    gang_s11 = torch.complex((SS.real) * (s11.real), (SS.imag) * (s11.imag))
    gang_s22 = torch.complex((SS.real) * (s22.real), (SS.imag) * (s22.imag))
    gang_s33 = torch.complex((SS.real) * (s33.real), (SS.imag) * (s33.imag))
    gang_s23 = torch.complex((SS.real) * (s23.real), (SS.imag) * (s23.imag))
    gang_s13 = torch.complex((SS.real) * (s13.real), (SS.imag) * (s13.imag))
    gang_s12 = torch.complex((SS.real) * (s12.real), (SS.imag) * (s12.imag))
    # sub-grid scale stress
    tau11 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s11).real
    tau22 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s22).real
    tau33 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s33).real
    tau23 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s23).real
    tau13 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s13).real
    tau12 = (-2.0 * Cs_square * ((filter_size) ** 2) * gang_s12).real
    tau11_h = torch.fft.fftn(tau11[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau22_h = torch.fft.fftn(tau22[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau33_h = torch.fft.fftn(tau33[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau23_h = torch.fft.fftn(tau23[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau13_h = torch.fft.fftn(tau13[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau12_h = torch.fft.fftn(tau12[:, :, :, :, :], dim=[1, 2, 3], norm='forward')

    tauij_x_h = 1j * (kx * tau11_h + ky * tau12_h + kz * tau13_h)
    tauij_y_h = 1j * (kx * tau12_h + ky * tau22_h + kz * tau23_h)
    tauij_z_h = 1j * (kx * tau13_h + ky * tau23_h + kz * tau33_h)
    # Divergence
    div_fft = ux_dx_h + uy_dy_h + uz_dz_h
    divergence = np.real(torch.fft.ifftn(div_fft, dim=[1, 2, 3]))

    Gx = uy * w_z - uz * w_y
    Gy = uz * w_x - ux * w_z
    Gz = ux * w_y - uy * w_x
    # cubic deliasing
    def apply_cube_dealiasing(G):
        device = G.device
        batch, X, Y, Z, dim, T = G.shape
        # Calculate the frequency threshold
        kx_max = X // 2
        ky_max = Y // 2
        kz_max = Z // 2
        kx_threshold = 2 * kx_max // 3
        ky_threshold = 2 * ky_max // 3
        kz_threshold = 2 * kz_max // 3
        # Generate frequency index
        kx = torch.fft.fftfreq(X, device=device) * X
        ky = torch.fft.fftfreq(Y, device=device) * Y
        kz = torch.fft.fftfreq(Z, device=device) * Z
        kx_mask = torch.abs(kx) > kx_threshold
        ky_mask = torch.abs(ky) > ky_threshold
        kz_mask = torch.abs(kz) > kz_threshold
        # Create a mask
        mask = torch.logical_or(kx_mask[:, None, None],
                                torch.logical_or(ky_mask[None, :, None], kz_mask[None, None, :])).to(device)
        # Apply the mask to each spectral dimension of G
        for b in range(batch):
            for d in range(dim):
                for t in range(T):
                    G[b, :, :, :, d, t] = torch.where(mask,
                                                      torch.complex(torch.tensor(0.0, device=device),
                                                                    torch.tensor(0.0, device=device)),
                                                      G[b, :, :, :, d, t])
        return G

    # Convection term
    Gx_h = torch.fft.fftn(Gx, dim=[1, 2, 3], norm='forward')
    Gy_h = torch.fft.fftn(Gy, dim=[1, 2, 3], norm='forward')
    Gz_h = torch.fft.fftn(Gz, dim=[1, 2, 3], norm='forward')
    # Deliasing
    Gx_h = apply_cube_dealiasing(Gx_h)
    Gy_h = apply_cube_dealiasing(Gy_h)
    Gz_h = apply_cube_dealiasing(Gz_h)
    # Correct SGS stress
    Gx_h = Gx_h - tauij_x_h
    Gy_h = Gy_h - tauij_y_h
    Gz_h = Gz_h - tauij_z_h

    right_term_x_h = Gx_h - kx * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_y_h = Gy_h - ky * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_z_h = Gz_h - kz * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_x = torch.fft.ifftn(right_term_x_h, dim=[1, 2, 3], norm='forward').real
    right_term_y = torch.fft.ifftn(right_term_y_h, dim=[1, 2, 3], norm='forward').real
    right_term_z = torch.fft.ifftn(right_term_z_h, dim=[1, 2, 3], norm='forward').real
    right_term = torch.cat((right_term_x, right_term_y, right_term_z), dim=-2)
    # viscosity term
    viscous_term_x_h = v * lap * ux_h
    viscous_term_x = torch.fft.ifftn(viscous_term_x_h, dim=[1, 2, 3], norm='forward').real
    viscous_term_y_h = v * lap * uy_h
    viscous_term_y = torch.fft.ifftn(viscous_term_y_h, dim=[1, 2, 3], norm='forward').real
    viscous_term_z_h = v * lap * uz_h
    viscous_term_z = torch.fft.ifftn(viscous_term_z_h, dim=[1, 2, 3], norm='forward').real
    viscous_term = torch.cat((viscous_term_x, viscous_term_y, viscous_term_z), dim=-2)
    ## first order
    # u_t = (u[:, :, :, :, :, 1:] - u[:, :, :, :, :, :-1]) /(t_interval/(nt-1))
    # u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 1)
    # f_loss = u_t - (right_term - viscous_term)[..., 1:]
    ## second order
    u_t = (u[:, :, :, :, :, 2:] - u[:, :, :, :, :, :-2]) /(2*(t_interval/(nt-1)))
    u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 2)
    f_loss = u_t - (right_term - viscous_term)[..., 1:-1]
    ## fourth order
    # u_t = (-u[:, :, :, :, :, 4:] + 8 * u[:, :, :, :, :, 3:-1] - 8 * u[:, :, :, :, :, 1:-3] + u[:, :, :, :, :, :-4]) / (
    #             12 * (t_interval/(nt-1)))
    # u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 4)
    # f_loss = u_t + (right_term + viscous_term)[..., 2:-2]

    return divergence,f_loss

def TML_Spectral_SM3d(u, v=1/40, Cs=0.001, t_interval=1.0): # u:[bc, S, S, S,  3, T],v=0.002,t_interval=2.0
    batchsize, nx, ny, nz, nc, nt  = u.shape
    device = u.device
    filter_size = (8*np.pi)/32
    Lx, Ly, Lz = 4 * np.pi, 8 * np.pi, 8 * np.pi
    # Velocity component
    ux = u[:, :, :, :, 0, :]
    uy = u[:, :, :, :, 1, :]
    uz = u[:, :, :, :, 2, :]
    ux = ux.reshape(batchsize, nx, ny, nz, 1, nt)
    uy = uy.reshape(batchsize, nx, ny, nz, 1, nt)
    uz = uz.reshape(batchsize, nx, ny, nz, 1, nt)
    ux_h = torch.fft.fftn(ux, dim=[1, 2, 3], norm='forward')
    uy_h = torch.fft.fftn(uy, dim=[1, 2, 3], norm='forward')
    uz_h = torch.fft.fftn(uz, dim=[1, 2, 3], norm='forward')
    # Define wave numbers in x, y, and z directions
    kx = torch.fft.fftfreq(nx, d=Lx / nx).to(device)
    ky = torch.fft.fftfreq(ny, d=Ly / ny).to(device)
    kz = torch.fft.fftfreq(nz, d=Lz / nz).to(device)
    kx = kx * 2 * np.pi
    ky = ky * 2 * np.pi
    kz = kz * 2 * np.pi
    kx = kx.view(1,nx, 1, 1, 1, 1).repeat(batchsize,1, ny, nz, 1, nt)
    ky = ky.view(1,1, ny, 1, 1, 1).repeat(batchsize,nx, 1, nz, 1, nt)
    kz = kz.view(1,1, 1, nz, 1, 1).repeat(batchsize,nx, ny, 1, 1, nt)
    lap = kx ** 2 + ky ** 2 + kz ** 2
    lap[:,0, 0, 0, :, :] = 1e-5
    # Vorticity component
    w_x_h = 1j * (ky * uz_h - kz * uy_h)
    w_y_h = 1j * (kz * ux_h - kx * uz_h)
    w_z_h = 1j * (kx * uy_h - ky * ux_h)
    w_x = torch.fft.ifftn(w_x_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward').real
    w_y = torch.fft.ifftn(w_y_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward').real
    w_z = torch.fft.ifftn(w_z_h[:, :, :, :, :], dim=[1, 2, 3], norm='forward').real
    # Compute derivatives in Fourier space
    ux_dx_h = 1j * kx * ux_h
    uy_dy_h = 1j * ky * uy_h
    uz_dz_h = 1j * kz * uz_h
    # Strain rate
    s11_h = ux_dx_h
    s22_h = uy_dy_h
    s33_h = uz_dz_h
    s23_h = 0.5 * 1j * (kz * uy_h + ky * uz_h)
    s13_h = 0.5 * 1j * (kx * uz_h + kz * ux_h)
    s12_h = 0.5 * 1j * (ky * ux_h + kx * uy_h)
    s11 = torch.fft.ifftn(s11_h, dim=[1, 2, 3], norm='forward').real
    s22 = torch.fft.ifftn(s22_h, dim=[1, 2, 3], norm='forward').real
    s33 = torch.fft.ifftn(s33_h, dim=[1, 2, 3], norm='forward').real
    s23 = torch.fft.ifftn(s23_h, dim=[1, 2, 3], norm='forward').real
    s13 = torch.fft.ifftn(s13_h, dim=[1, 2, 3], norm='forward').real
    s12 = torch.fft.ifftn(s12_h, dim=[1, 2, 3], norm='forward').real
    # Characteristic strain rate
    SS = 2.0 * (((s11.real) * (s11.real)
                 + (s22.real) * (s22.real)
                 + (s33.real) * (s33.real)
                 + 2.0 * (s23.real) * (s23.real)
                 + 2.0 * (s13.real) * (s13.real)
                 + 2.0 * (s12.real) * (s12.real)))
    SS = torch.sqrt(SS.real)

    gang_s11 = (SS.real) * (s11.real)
    gang_s22 = (SS.real) * (s22.real)
    gang_s33 = (SS.real) * (s33.real)
    gang_s23 = (SS.real) * (s23.real)
    gang_s13 = (SS.real) * (s13.real)
    gang_s12 = (SS.real) * (s12.real)
    # Sub-grid scale stress
    tau11 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s11)
    tau22 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s22)
    tau33 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s33)
    tau23 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s23)
    tau13 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s13)
    tau12 = (-2.0 * Cs * ((filter_size) ** 2) * gang_s12)
    tau11_h = torch.fft.fftn(tau11[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau22_h = torch.fft.fftn(tau22[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau33_h = torch.fft.fftn(tau33[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau23_h = torch.fft.fftn(tau23[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau13_h = torch.fft.fftn(tau13[:, :, :, :, :], dim=[1, 2, 3], norm='forward')
    tau12_h = torch.fft.fftn(tau12[:, :, :, :, :], dim=[1, 2, 3], norm='forward')

    tauij_x_h = 1j * (kx * tau11_h + ky * tau12_h + kz * tau13_h)
    tauij_y_h = 1j * (kx * tau12_h + ky * tau22_h + kz * tau23_h)
    tauij_z_h = 1j * (kx * tau13_h + ky * tau23_h + kz * tau33_h)
    # Divergence
    div_fft = ux_dx_h + uy_dy_h + uz_dz_h
    divergence = np.real(torch.fft.ifftn(div_fft, dim=[1, 2, 3], norm='forward'))

    Gx = uy * w_z - uz * w_y
    Gy = uz * w_x - ux * w_z
    Gz = ux * w_y - uy * w_x
    # Cubic deliasing
    def apply_dealiasing_correctly(G):
        device = G.device
        # Calculate the frequency threshold
        batch, X, Y, Z, dim, T = G.shape
        # Generate frequency index
        kx_max = X / (2 * 2)
        ky_max = Y / (2 * 4)
        kz_max = Z / (2 * 4)
        kx_threshold = 2 * kx_max / 3
        ky_threshold = 2 * ky_max / 3
        kz_threshold = 2 * kz_max / 3
        kx = torch.fft.fftfreq(X, d=Lx / X, device=device) * 2 * np.pi
        ky = torch.fft.fftfreq(Y, d=Ly / Y, device=device) * 2 * np.pi
        kz = torch.fft.fftfreq(Z, d=Lz / Z, device=device) * 2 * np.pi
        kx_mask = torch.abs(kx) > kx_threshold
        ky_mask = torch.abs(ky) > ky_threshold
        kz_mask = torch.abs(kz) > kz_threshold
        # Create a mask
        mask = torch.logical_or(kx_mask[:, None, None],
                                torch.logical_or(ky_mask[None, :, None], kz_mask[None, None, :])).to(device)
        # Apply the mask to each spectral dimension of G
        for b in range(batch):
            for d in range(dim):
                for t in range(T):
                    G[b, :, :, :, d, t] = torch.where(mask,
                                                      torch.complex(torch.tensor(0.0, device=device),
                                                                    torch.tensor(0.0, device=device)),
                                                      G[b, :, :, :, d, t])
        return G

    # Convection term
    Gx_h = torch.fft.fftn(Gx, dim=[1, 2, 3], norm='forward')
    Gy_h = torch.fft.fftn(Gy, dim=[1, 2, 3], norm='forward')
    Gz_h = torch.fft.fftn(Gz, dim=[1, 2, 3], norm='forward')
    # Deliasing
    Gx_h = apply_dealiasing_correctly(Gx_h)
    Gy_h = apply_dealiasing_correctly(Gy_h)
    Gz_h = apply_dealiasing_correctly(Gz_h)
    # Correct SGS stress
    Gx_h = Gx_h - tauij_x_h
    Gy_h = Gy_h - tauij_y_h
    Gz_h = Gz_h - tauij_z_h

    right_term_x_h = Gx_h - kx * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_y_h = Gy_h - ky * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_z_h = Gz_h - kz * (kx * Gx_h + ky * Gy_h + kz * Gz_h) / lap
    right_term_x = torch.fft.ifftn(right_term_x_h, dim=[1, 2, 3], norm='forward').real
    right_term_y = torch.fft.ifftn(right_term_y_h, dim=[1, 2, 3], norm='forward').real
    right_term_z = torch.fft.ifftn(right_term_z_h, dim=[1, 2, 3], norm='forward').real
    right_term = torch.cat((right_term_x, right_term_y, right_term_z), dim=-2)
    # viscosity term
    viscous_term_x_h = v * lap * ux_h
    viscous_term_x = torch.fft.ifftn(viscous_term_x_h, dim=[1, 2, 3], norm='forward').real
    viscous_term_y_h = v * lap * uy_h
    viscous_term_y = torch.fft.ifftn(viscous_term_y_h, dim=[1, 2, 3], norm='forward').real
    viscous_term_z_h = v * lap * uz_h
    viscous_term_z = torch.fft.ifftn(viscous_term_z_h, dim=[1, 2, 3], norm='forward').real
    viscous_term = torch.cat((viscous_term_x, viscous_term_y, viscous_term_z), dim=-2)

    ## first order
    # u_t = (u[:, :, :, :, :, 1:] - u[:, :, :, :, :, :-1]) /(t_interval/(nt-1))
    # u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 1)
    ## second order
    u_t = (u[:, :, :, :, :, 2:] - u[:, :, :, :, :, :-2]) /(2*(t_interval/(nt-1)))
    u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 2)
    f_loss = u_t - (right_term - viscous_term)[..., 1:-1]
    ## fourth order
    # u_t = (-u[:, :, :, :, :, 4:] + 8 * u[:, :, :, :, :, 3:-1] - 8 * u[:, :, :, :, :, 1:-3] + u[:, :, :, :, :, :-4]) / (
    #             12 * (t_interval/(nt-1)))
    # u_t = u_t.reshape(batchsize, nx, ny, nz, 3, nt - 4)
    # f_loss = u_t + (right_term + viscous_term)[..., 2:-2]

    return divergence,f_loss


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def PINO_loss3d(u, v=1/40, Cs_square=0.1, t_interval=1.0,flow_name ='DHIT'):

    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nz = u.size(3)
    nt = u.size(5)
    u= u.reshape(batchsize, nx, ny, nz, 3, nt)
    if flow_name == 'DHIT':
        divergence,f_loss = DHIT_Spectral_SM3d(u, v,Cs_square, t_interval)
    if flow_name == 'TML':
        divergence, f_loss = TML_Spectral_SM3d(u, v, Cs_square, t_interval)

    div = torch.zeros(divergence.shape, device=u.device)
    f = torch.zeros(f_loss.shape, device=u.device)
    loss_f = F.mse_loss(f_loss, f) + F.mse_loss(divergence, div)

    return  loss_f

