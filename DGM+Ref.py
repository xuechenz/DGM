import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, math, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      
from torch.distributions import Beta

T, r, sigma   = 1.0, 0.02, 0.05
S_Max, S0     = 200.0, 100.0
w_ic = 1e-6
density_file  = "gbm_density_hist.npz"
S_window      = 100.0
N_REF, batch_ref = 20_000, 1024

d        = np.load(density_file)
t_grid   = torch.tensor(d["t_list"],    dtype=torch.float32)
S_grid   = torch.tensor(d["S_centers"], dtype=torch.float32)
pdf      = torch.tensor(d["density"],   dtype=torch.float32)

mask     = (S_grid >= S0-S_window) & (S_grid <= S0+S_window)
TT, SS   = torch.meshgrid(t_grid, S_grid[mask], indexing='ij')
XY_all   = torch.stack([TT.flatten(), SS.flatten()], 1)
v_all    = pdf[:, mask].flatten().unsqueeze(1)

perm     = torch.randperm(XY_all.size(0))[:N_REF]
XY_ref_cpu, v_ref_cpu = XY_all[perm], v_all[perm]
NUM_REF  = XY_ref_cpu.size(0)
print(f"Loaded {NUM_REF} reference points")


def sampler_gbm(batch, device,
                S0=S0, mu=r, vol=sigma, T=T,
                t_min=1e-4):                     
    t = (t_min + torch.rand(batch, 1, device=device) * (T - t_min))

    z = torch.randn_like(t)                    
    S = S0 * torch.exp((mu - 0.5 * vol**2) * t +
                       vol * torch.sqrt(t) * z)

    return torch.hstack([t, S])



def fp_residual(net, X, mu=r, vol=sigma):
    X = X.clone().detach().requires_grad_(True)
    t, S = X[:, :1], X[:, 1:2]

    p = net(X)                      

    grad_p = torch.autograd.grad(
        p, X, torch.ones_like(p), create_graph=True
    )[0]
    p_t = grad_p[:, :1]               

    flux1   = mu * S * p
    grad_f1 = torch.autograd.grad(
        flux1, X, torch.ones_like(flux1), create_graph=True
    )[0]
    flux1_S = grad_f1[:, 1:2]

    flux2      = 0.5 * vol**2 * S**2 * p
    grad_f2    = torch.autograd.grad(
        flux2, X, torch.ones_like(flux2), create_graph=True
    )[0]
    flux2_S    = grad_f2[:, 1:2]
    grad_f2_S  = torch.autograd.grad(
        flux2_S, X, torch.ones_like(flux2_S), create_graph=True
    )[0]
    flux2_SS = grad_f2_S[:, 1:2]

    return p_t + flux1_S - flux2_SS


class DGMBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.U_z = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W_z = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.U_g = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W_g = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.U_r = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W_r = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.U_h = nn.Linear(in_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.act = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, s_prev, s_first):
        z = self.act(self.U_z(x) + self.W_z(s_prev))
        g = self.act(self.U_g(x) + self.W_g(s_first))
        r = self.act(self.U_r(x) + self.W_r(s_prev))
        h = self.act(self.U_h(x) + self.W_h(s_prev * r))

        s_next = (1.0 - g) * h + z * s_prev
        return s_next
class DGMNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        self.blocks = nn.ModuleList([
            DGMBlock(input_dim, hidden_dim) for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.act = nn.Tanh()

    def forward(self, x):
        s = self.act(self.input_layer(x))
        s_first = s

        for block in self.blocks:
            s = block(x, s, s_first)

        return torch.exp(self.output_layer(s))                 

def loss_fn(net, X_in, XY_mb, v_mb):
    loss_pde  = fp_residual(net, X_in).pow(2).mean()
    loss_ref  = (net(XY_mb) - v_mb).pow(2).mean()
    return loss_pde + loss_ref, loss_pde, loss_ref


if __name__ == "__main__":
    batch, epochs, lr = 4096, 10000, 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = DGMNet(input_dim=2, hidden_dim=128, n_layers=4).to(device)
    opt    = optim.Adam(net.parameters(), lr=lr)

    XY_ref = XY_ref_cpu.to(device)
    v_ref  = v_ref_cpu.to(device)

    for ep in range(1, epochs + 1):
        X_in = sampler(batch, device)
        sel  = torch.randint(0, NUM_REF, (batch_ref,), device=device)
        XY_mb, v_mb = XY_ref[sel], v_ref[sel]

        loss_tot, lpde, lref = loss_fn(net, X_in, XY_mb, v_mb)
        opt.zero_grad(); loss_tot.backward(); opt.step()

        if ep % 500 == 0:
            print(f"Ep{ep:5d} | tot={loss_tot:.2e} | "
                  f"PDE={lpde:.2e} | REF={lref:.2e}")

    torch.save(net.state_dict(), "DGM_pde_ic_ref.pth")
