import torch, torch.nn as nn, torch.optim as optim
import math, time

T, r, sigma = 1.0, 0.02, 0.05
S_Max, S0   = 200.0, 100.0      

def sampler(batch, eps, device, use_gbm=True):
    # Inner Point from log normal
    t = torch.rand(batch,1, device=device) * T
    if use_gbm:
        z = torch.randn_like(t)
        S = S0 * torch.exp((r - 0.5 * sigma**2) * t + sigma * torch.sqrt(t) * z)
    else:
        S = torch.rand(batch,1, device=device) * S_Max
    X_in = torch.cat([t, S], dim=1)

    # Initial Points from Gaussian Approx to Delta function 
    S_ic = S0 + eps * torch.randn(batch,1, device=device)
    X_ic = torch.cat([torch.zeros_like(S_ic), S_ic], dim=1)

    # Boundary points near 0 and S_Max the density should be 0
    t_b = torch.rand(2*batch,1, device=device) * T
    S_b = torch.cat([torch.zeros(batch,1, device=device),
                     torch.full((batch,1), S_Max, device=device)], dim=0)
    X_b = torch.cat([t_b, S_b], dim=1)

    # Mass Sampling for the integration = 1
    S_mass = torch.rand(batch,1, device=device) * S_Max
    return X_in, X_ic, X_b, S_mass

def fp_residual(net, X):
    X.requires_grad_(True)
    p    = net(X)
    grad = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0]
    p_t, p_S = grad[:,0:1], grad[:,1:2]
    p_SS = torch.autograd.grad(p_S, X, torch.ones_like(p_S), create_graph=True)[0][:,1:2]

    S = X[:,1:2]
    res = p_t - ((sigma**2 - r)*p + (2*sigma**2*S - r*S)*p_S + 0.5*sigma**2*S**2*p_SS)
    return res

def loss_fn(net, X_in, X_ic, X_b, S_mass, eps):
    # PDE residual
    pde_res = fp_residual(net, X_in)
    loss_pde = (pde_res ** 2).mean()

    # Initial condition (narrow Gaussian: Delta Function Approxi)
    p_pred = net(X_ic)
    p_true = torch.exp(-(X_ic[:, 1:2] - S0) ** 2 / (2 * eps**2)) \
             / (eps * math.sqrt(2 * math.pi))
    loss_ic = ((p_pred - p_true) ** 2).mean()

    # Density near S=0 and S=S_max should equal 0
    loss_bc = (net(X_b) ** 2).mean()

    # Mass conservation  integration should equal to 1
    t_mass = torch.rand_like(S_mass) * T
    X_mass = torch.cat([t_mass, S_mass], dim=1)
    loss_mass = ((net(X_mass).mean() * S_Max - 1.0) ** 2)

    total = loss_pde + loss_ic + loss_bc + loss_mass
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

        return self.softplus(self.output_layer(s))
  

if __name__ == "__main__":
    batch, epochs, lr = 2048, 8000, 1e-3
    eps0, eps_min = 0.5, 0.05         
    log_every = 500                    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DGMNet(hidden_dim=32).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    start = time.time()
    for ep in range(1, epochs + 1):
        eps = max(eps_min, eps0 * (0.999 ** ep))       
        X_in, X_ic, X_b, S_mass = sampler(batch, eps, device)

        loss_tot, loss_pde, loss_ic, loss_bc, loss_mass = \
            loss_fn(net, X_in, X_ic, X_b, S_mass, eps)

        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        if ep % log_every == 0:
            print(f"Epoch {ep:5d} | total={loss_tot.item():.3e} | "
                  f"PDE={loss_pde.item():.3e} | IC={loss_ic.item():.3e} | "
                  f"BC={loss_bc.item():.3e} | Mass={loss_mass.item():.3e}")

    print(f"Training finished in {time.time() - start:.1f} s")
    torch.save(net.state_dict(), "DGM_forward_simple.pth")

