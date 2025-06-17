import numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import tqdm, math, random

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(20240617)
np.random.seed(20240617)
random.seed(20240617)

S0, r, sigma = 100.0, 0.02, 0.05
DT          = 1/504         
HIDDEN      = 64
N_EPOCH_REF = 2000           
N_EPOCH_FIN = 1000           
LR          = 3e-3
W_PDE       = 5.0          
W_REF       = 1.0

d       = np.load("gbm_density_hist.npz")
t_list  = torch.tensor(d["t_list"],    dtype=torch.float32)       
S_grid  = torch.tensor(d["S_centers"], dtype=torch.float32)       
pdf     = torch.tensor(d["density" ],  dtype=torch.float32)       
bin_w   = float(S_grid[1]-S_grid[0])                             

N_t, N_S = pdf.shape
assert N_t == len(t_list)

class SliceMLP(nn.Module):
    def __init__(self, hidden=HIDDEN, n_layers=3, t_value=0.0, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.t_const = nn.Parameter(torch.tensor([[t_value]]),
                                    requires_grad=False)

        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.body = nn.Sequential(*layers)
        self.out  = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S_in):               
        t_in = self.t_const.expand_as(S_in)
        h    = self.body(torch.cat([t_in, S_in], dim=1))
        return torch.nn.functional.softplus(self.out(h)) + self.eps 

    def pdf(self, S_query):
        return self.forward(S_query)

models = nn.ModuleList([SliceMLP(t_value=float(t)) for t in t_list]).to(device)

def mse_ref(model, S_batch, p_target):
    p_pred = model.pdf(S_batch)
    return (p_pred - p_target).pow(2).mean()

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

optimizer = optim.Adam(models.parameters(), lr=LR)

def train_ref_only():
    loader_S = S_grid.unsqueeze(1).to(device)       

    for i in range(N_t):                           
        model   = models[i]
        p_target= pdf[i].unsqueeze(1).to(device)

        optim_i = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(N_EPOCH_REF):
            loss = mse_ref(model, loader_S, p_target)
            optim_i.zero_grad()
            loss.backward()
            optim_i.step()

            if (epoch+1) % 500 == 0:
                tqdm.tqdm.write(f"[Ref-chain] slice {i}/{N_t-1} "
                                f"epoch {epoch+1}, loss={loss.item():.4e}")

        if i < N_t-1:
            models[i+1].load_state_dict(model.state_dict())

def train_with_pde():
    loader_S = S_grid.unsqueeze(1).to(device)
    dt = float(t_list[1]-t_list[0])                 
    for epoch in tqdm.trange(N_EPOCH_FIN, desc="Stage-2 PDE"):
        total_loss = 0.
        for i in range(N_t-1):
            mdl_i, mdl_ip1 = models[i], models[i+1]

            p_trg_i   = pdf[i  ].unsqueeze(1).to(device)
            p_trg_ip1 = pdf[i+1].unsqueeze(1).to(device)
            ref_loss  = mse_ref(mdl_i, loader_S, p_trg_i) \
                       +mse_ref(mdl_ip1,loader_S, p_trg_ip1)

            with torch.enable_grad():
                S_batch = loader_S                 
                def net_pair(X):
                    S_in = X[:,1:2]
                    p_i   = mdl_i.pdf(S_in)
                    p_ip1 = mdl_ip1.pdf(S_in)
                    t0 = float(t_list[i])
                    return p_i + (X[:,0:1]-t0)*(p_ip1-p_i)/dt

                t_tensor = torch.full_like(S_batch, float(t_list[i]))
                X_in     = torch.cat([t_tensor, S_batch], dim=1).detach()
                X_in.requires_grad_(True)

                pde_res  = fp_residual(net_pair, X_in, mu=r, vol=sigma)
                pde_loss = (pde_res**2).mean()

            total_loss += W_REF*ref_loss + W_PDE*pde_loss

        total_loss /= (N_t-1)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch+1) % 500 == 0:
            tqdm.tqdm.write(f"[PDE] epoch {epoch+1}, loss={total_loss.item():.4e}")

if __name__ == "__main__":
    print(">> Stage-1: only Ref-loss")
    train_ref_only()
    print(">> Stage-2: add PDE-loss (finite-diff ∂t)")
    train_with_pde()

    out_dir = Path("slice_mlps")
    out_dir.mkdir(exist_ok=True)
    for t, mdl in zip(t_list, models):
        torch.save(mdl.state_dict(), out_dir / f"mlp_t{float(t):.3f}.pt")
    print("All slice-MLPs saved in ./slice_mlps/")
