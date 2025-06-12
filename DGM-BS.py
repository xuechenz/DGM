import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

def sampler(batch, T, S_Max,
            R_Min, R_Max, SIG_Min, SIG_Max,
            K_Min, K_Max, device):
    # Internal points: (t, S, r, sigma, K) uniformly sampled in the domain
    t   = torch.rand(batch,1, device=device)*T
    S   = torch.rand(batch,1, device=device)*S_Max
    r   = torch.rand(batch,1, device=device)*(R_Max-R_Min)+R_Min
    sig = torch.rand(batch,1, device=device)*(SIG_Max-SIG_Min)+SIG_Min
    K_  = torch.rand(batch,1, device=device)*(K_Max-K_Min)+K_Min
    X_in = torch.cat([t, S, r, sig, K_], dim=1)

    # Terminal points: set t = T
    X_T = X_in.clone()
    X_T[:,0:1] = T

    # Boundary condition points: S = 0 and S = S_Max
    t_b   = torch.rand_like(t)
    r_b   = torch.rand_like(r)*(R_Max-R_Min)+R_Min
    sig_b = torch.rand_like(sig)*(SIG_Max-SIG_Min)+SIG_Min
    K_b   = torch.rand_like(K_)*(K_Max-K_Min)+K_Min
    X_b0  = torch.cat([t_b, torch.zeros_like(S),      r_b, sig_b, K_b], dim=1)
    X_b1  = torch.cat([t_b, torch.full_like(S,S_Max), r_b, sig_b, K_b], dim=1)
    X_b   = torch.cat([X_b0, X_b1], dim=0)
    return X_in, X_T, X_b

def pde_residual(net, X, delta=1e-3, mc_samples=2):
    mc_samples = max(2, mc_samples)
    if mc_samples % 2 != 0:
        mc_samples += 1 

    X = X.clone().requires_grad_(True)
    u = net(X)
    grad = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_t, u_S = grad[:, 0:1], grad[:, 1:2] 

    S   = X[:, 1:2]
    r   = X[:, 2:3]
    sig = X[:, 3:4]

    # Antithetic noise: (+eps, -eps)
    half = mc_samples // 2
    eps = torch.randn(half, *u_S.shape, device=X.device) * math.sqrt(delta)
    eps = torch.cat([eps, -eps], dim=0)
    eps = eps + 1e-8 * eps.sign()  # avoid zero

    # S_pert = S + sig * S * eps
    S_pert = (S + sig * S * eps).detach().requires_grad_(True)

    X_pert = X.repeat(mc_samples, 1)
    X_pert[:, 1:2] = S_pert.reshape(-1, 1)

    u_pert = net(X_pert)
    grad_pert = torch.autograd.grad(u_pert, X_pert, torch.ones_like(u_pert), create_graph=True)[0]
    u_S_pert = grad_pert[:, 1:2]

    # (du/dS_pert - du/dS) / (sigma * S * eps)
    eps_flat = eps.reshape(-1, 1)
    diff = (u_S_pert - u_S.repeat(mc_samples, 1)) / (sig.repeat(mc_samples, 1) * S.repeat(mc_samples, 1) * eps_flat)
    u_SS_mc = diff.mean(dim=0)

    return u_t + 0.5 * sig**2 * S**2 * u_SS_mc + r * S * u_S - r * u




def loss_fn(net, X_in, X_T, X_b, S_Max, T):
    res = pde_residual(net, X_in)
    loss_pde = torch.mean(res**2)

    u_T    = net(X_T)
    payoff = torch.relu(X_T[:,1:2] - X_T[:,4:5])  # S - K
    loss_T = torch.mean((u_T - payoff)**2)

    u_b    = net(X_b)  # Boundary prediction
    t_b, S_b, r_b, K_b = X_b[:,0:1], X_b[:,1:2], X_b[:,2:3], X_b[:,4:5]
    N = X_b.shape[0]//2
    bc0 = u_b[:N]
    bc1 = u_b[N:] - (S_Max - K_b[N:] * torch.exp(-r_b[N:] * (T - t_b[N:])))  # discounted intrinsic
    loss_b = torch.mean(bc0**2) + torch.mean(bc1**2)

    return loss_pde+loss_T+loss_b, loss_pde, loss_T, loss_b

# DGM Blocks
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
#DGM Net
class DGMNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, n_layers=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)

        self.blocks = nn.ModuleList([
            DGMBlock(input_dim, hidden_dim) for _ in range(n_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        self.act = nn.Tanh()

    def forward(self, x):
        s = self.act(self.input_layer(x))
        s_first = s

        for block in self.blocks:
            s = block(x, s, s_first)

        return self.output_layer(s)

if __name__ == "__main__":
    T, K, R, SIGMA, S_Max = 1.0, 100.0, 0.02, 0.05, 200.0
    batch, epochs, lr = 2048, 10000, 1e-3
    R_Min, R_Max = 0.0, 0.1
    K_Min, K_Max = 95.0, 105.0
    Sig_min, Sig_max = 0.05, 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization & training
    net = DGMNet(hidden_dim=32).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    start_time = time.time()
    for epoch in range(1, epochs+1):
        X_in, X_T, X_b = sampler(batch, T, S_Max,
                         R_Min, R_Max, Sig_min, Sig_max,
                         K_Min, K_Max, device)
        loss, l_pde, l_T, l_b = loss_fn(net, X_in, X_T, X_b, S_Max, T)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:>4}: total={loss.item():.2e}, PDE={l_pde:.2e}, "
                  f"T={l_T:.2e}, B={l_b:.2e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    filename = f"DGM_{lr}_{batch}_{epochs}.pth"
    torch.save(net.state_dict(), filename)
    print(f"Model saved to {filename}")
