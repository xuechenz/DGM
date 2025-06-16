import math, time
import torch, torch.nn as nn, torch.optim as optim

T, r, sigma = 1.0, 0.02, 0.05          
S0, S_max   = 100.0, 200.0             
mu_tilde    = r                        
a = mu_tilde - 0.5 * sigma ** 2        
y_min = -10.0                         
y_max = math.log(S_max / S0)          


def sampler(batch, eps, device, tail_frac=0.3):
    # inner points 
    batch_gbm = int(batch * (1 - tail_frac))
    t = torch.rand(batch_gbm, 1, device=device) * T
    z = torch.randn_like(t)
    S = S0 * torch.exp((r - 0.5 * sigma**2) * t + sigma * torch.sqrt(t) * z)
    y = torch.log(S / S0)

    # tail points 
    batch_tail = batch - batch_gbm
    t_tail = torch.rand(batch_tail, 1, device=device) * T
    y_tail = y_min + (y_max - y_min) * torch.rand(batch_tail, 1, device=device)

    # concat
    t_in = torch.vstack([t, t_tail])
    y_in = torch.vstack([y, y_tail])
    X_in = torch.hstack([t_in, y_in])

    # ---- initial, boundary, mass 同之前 ----
    y_ic  = eps * torch.randn(batch, 1, device=device)
    X_ic  = torch.hstack([torch.zeros_like(y_ic), y_ic])

    t_b = torch.rand(2*batch, 1, device=device) * T
    y_b = torch.vstack([
        torch.full((batch,1), y_min, device=device),
        torch.full((batch,1), y_max, device=device)
    ])
    X_bc = torch.hstack([t_b, y_b])

    y_mass = y_min + (y_max - y_min) * torch.rand(batch, 1, device=device)
    return X_in, X_ic, X_bc, y_mass


# PDE Residual

def fp_residual(net: nn.Module, X: torch.Tensor):
    """Return F = q_t + a q_y − ½ σ² q_yy"""
    X.requires_grad_(True)
    q = net(X)

    grad  = torch.autograd.grad(q, X, torch.ones_like(q), create_graph=True)[0]
    q_t   = grad[:, 0:1]
    q_y   = grad[:, 1:2]
    q_yy  = torch.autograd.grad(q_y, X, torch.ones_like(q_y), create_graph=True)[0][:, 1:2]

    return q_t + a * q_y - 0.5 * sigma ** 2 * q_yy

def loss_fn(net: nn.Module,
            X_in: torch.Tensor,
            X_ic: torch.Tensor,
            X_bc: torch.Tensor,
            y_mass: torch.Tensor,
            eps: float):
    # PDE residual
    res      = fp_residual(net, X_in)
    loss_pde = (res ** 2).mean()

    # Initial condition 
    q_pred0  = net(X_ic)
    q_true0  = torch.exp(-(X_ic[:, 1:2] ** 2) / (2 * eps ** 2)) / (eps * math.sqrt(2 * math.pi))
    loss_ic  = ((q_pred0 - q_true0) ** 2).mean()

    # Boundary
    loss_bc  = (net(X_bc) ** 2).mean()

    # Mass 
    t_mass   = torch.rand_like(y_mass) * T
    X_mass   = torch.hstack([t_mass, y_mass])
    loss_mass = ((net(X_mass).mean() * (y_max - y_min) - 1.0) ** 2)

    total = loss_pde + loss_ic + 0.1 * loss_bc + loss_mass
    return total, loss_pde, loss_ic, loss_bc, loss_mass

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


# train
if __name__ == "__main__":
    batch_sz, epochs, lr = 2048, 15000, 2e-5
    eps0, eps_min        = 0.5, 0.02         
    log_step             = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = DGMNet(hidden_dim=128, n_layers=5).to(device)
    optim_ = optim.Adam(net.parameters(), lr=lr)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        eps = max(eps_min, eps0 * (0.999 ** ep))
        X_in, X_ic, X_bc, y_mass = sampler(batch_sz, eps, device)

        loss_tot, lpde, lic, lbc, lmass = loss_fn(net, X_in, X_ic, X_bc, y_mass, eps)
        optim_.zero_grad(); loss_tot.backward(); optim_.step()

        if ep % log_step == 0:
            print(f"Ep {ep:6d} | total={loss_tot.item():.3e} | PDE={lpde.item():.1e} | "
                  f"IC={lic.item():.1e} | BC={lbc.item():.1e} | Mass={lmass.item():.1e} | eps={eps:.3f}")

    print(f"Training done in {time.time() - t0:.1f}s — saving model…")
    torch.save(net.state_dict(), "DGM_forward_log.pth")
