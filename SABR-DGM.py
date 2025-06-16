import math, time
import torch, torch.nn as nn, torch.optim as optim

def sampler_sabr(batch, T,
                 F_min, F_max,
                 A_min, A_max,
                 K_min, K_max,
                 beta_min, beta_max,
                 nu_min,   nu_max,
                 rho_min,  rho_max,
                 device):
    # Inner Points
    t  = torch.rand(batch,1, device=device) * T
    F  = F_min + (F_max-F_min) * torch.rand_like(t)
    A  = A_min + (A_max-A_min) * torch.rand_like(t)
    K  = K_min + (K_max-K_min) * torch.rand_like(t)
    β  = beta_min + (beta_max-beta_min) * torch.rand_like(t)
    ν  = nu_min   + (nu_max-nu_min)     * torch.rand_like(t)
    ρ  = rho_min  + (rho_max-rho_min)   * torch.rand_like(t)
    X_in = torch.cat([t,F,A,K,β,ν,ρ], dim=1)

    # term points
    X_T = X_in.clone();  X_T[:,0:1] = T

    # boundary points
    t_b  = torch.rand_like(t)
    A_b  = A_min + (A_max-A_min) * torch.rand_like(t)
    K_b  = K_min + (K_max-K_min) * torch.rand_like(t)
    β_b  = beta_min + (beta_max-beta_min) * torch.rand_like(t)
    ν_b  = nu_min   + (nu_max-nu_min)     * torch.rand_like(t)
    ρ_b  = rho_min  + (rho_max-rho_min)   * torch.rand_like(t)

    X_b0 = torch.cat([t_b, torch.zeros_like(F),      A_b, K_b, β_b, ν_b, ρ_b], dim=1)
    X_b1 = torch.cat([t_b, torch.full_like(F,F_max), A_b, K_b, β_b, ν_b, ρ_b], dim=1)
    X_b  = torch.cat([X_b0, X_b1], dim=0)
    return X_in, X_T, X_b


def pde_residual_sabr(net, X):
    X = X.clone().requires_grad_(True)
    u    = net(X)
    grad = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_t, u_F, u_A = grad[:,0:1], grad[:,1:2], grad[:,2:3]

    # second‑order terms
    u_F_grad = torch.autograd.grad(u_F, X, torch.ones_like(u_F), create_graph=True)[0]
    u_FF = u_F_grad[:,1:2]
    u_FA = u_F_grad[:,2:3]

    u_A_grad = torch.autograd.grad(u_A, X, torch.ones_like(u_A), create_graph=True)[0]
    u_AA = u_A_grad[:,2:3]

    F, A = X[:,1:2], X[:,2:3]
    β, ν, ρ = X[:,4:5], X[:,5:6], X[:,6:7]

    res = (u_t
           + 0.5*A**2 * F**(2*β) * u_FF
           + ρ*ν*A**2 * F**β   * u_FA
           + 0.5*ν**2 * A**2 * u_AA)
    return res


def loss_fn_sabr(net, X_in, X_T, X_b, F_max):
    res      = pde_residual_sabr(net, X_in)
    loss_pde = (res**2).mean()

    # terminal payoff (F-K)^+
    payoff   = torch.relu(X_T[:,1:2] - X_T[:,3:4])
    loss_T   = ((net(X_T) - payoff)**2).mean()

    # boundaries in F
    u_b = net(X_b)
    N   = X_b.shape[0]//2
    bc0 = u_b[:N]                      
    bc1 = u_b[N:] - (F_max - X_b[N:,3:4])  
    loss_b = (bc0**2).mean() + (bc1**2).mean()

    return loss_pde + loss_T + loss_b, loss_pde, loss_T, loss_b

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
    def __init__(self, input_dim=7, hidden_dim=64, n_layers=3):
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
    T = 1.0
    F_min, F_max = 60.0, 140.0
    A_min, A_max = 0.05, 0.15
    K_min, K_max = 95.0, 105.0

    beta_min, beta_max = 0.4, 1.0    
    nu_min,   nu_max   = 0.1, 0.6     
    rho_min,  rho_max  = -0.9, 0.9   

    batch, epochs, lr = 2048, 12000, 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DGMNet().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    t0 = time.time()
    for ep in range(1, epochs+1):
        X_in, X_T, X_b = sampler_sabr(batch, T,
                                      F_min, F_max,
                                      A_min, A_max,
                                      K_min, K_max,
                                      beta_min, beta_max,
                                      nu_min,   nu_max,
                                      rho_min,  rho_max,
                                      device)
        loss, lpde, lT, lb = loss_fn_sabr(net, X_in, X_T, X_b, F_max)
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 500 == 0:
            print(f"Epoch {ep:>5} | total={loss.item():.3e} | PDE={lpde:.2e} | T={lT:.2e} | B={lb:.2e}")

    print(f"Finished in {time.time()-t0:.1f}s")
    torch.save(net.state_dict(), "SABR_DGM_universal.pth")
    print("Model saved → SABR_DGM_universal.pth")
