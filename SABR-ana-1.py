# Deep‑BSDE vs. SABR analytical pricing curves
#
# ‑ For each fixed α ∈ {0.05, 0.10, 0.15} we sweep F from 60 → 140 and
#   plot the t = 0 option price given by:
#       • our trained Deep‑BSDE network (Y0_net)
#       • the Hagan (2002) SABR closed‑form approximation translated to a
#         Black–Scholes call price
#
# ‑ If the checkpoint “deepbsde_sabr.pth” is not found the script still
#   runs, drawing the analytical curve only (with a notice).

import os, math, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from torch.distributions.normal import Normal

# ---------- 1. SABR → BS helper ----------
def hagan_implied_vol(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    if abs(F - K) < eps:  # ATM
        fk_beta = F ** (1 - beta)
        term1 = alpha / fk_beta
        term2 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / fk_beta ** 2
        term3 = (rho * beta * nu * alpha) / (4 * fk_beta)
        term4 = (2 - 3 * rho ** 2) * nu ** 2 / 24
        return term1 * (1 + (term2 + term3 + term4) * T)

    logFK = math.log(F / K)
    FK = F * K
    fk_beta = FK ** ((1 - beta) / 2)
    z = (nu / alpha) * fk_beta * logFK
    x_z = math.log((math.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    A = alpha / fk_beta
    B = 1 + ((1 - beta) ** 2 / 24) * logFK ** 2 + ((1 - beta) ** 4 / 1920) * logFK ** 4
    C = 1 + (
        ((1 - beta) ** 2 * alpha ** 2) / (24 * FK ** (1 - beta))
        + (rho * beta * nu * alpha) / (4 * FK ** ((1 - beta) / 2))
        + (2 - 3 * rho ** 2) * nu ** 2 / 24
    ) * T
    return (A / B) * (z / x_z) * C

def bs_call(F, K, T, sigma):
    N = Normal(0.0, 1.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * N.cdf(torch.tensor(d1)) - K * N.cdf(torch.tensor(d2))

# ---------- 2. Deep‑BSDE network ----------
hidden, N = 64, 100
beta, nu, rho = 0.7, 0.4, -0.3
T, K = 1.0, 100.0
sqrt_dt = math.sqrt(T / N)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

class DeepBSDE_SABR(nn.Module):
    def __init__(self):
        super().__init__()
        self.u0_net = MLP(2, hidden, 1)
        self.z0_net = MLP(2, hidden, 2)
        self.z_nets = nn.ModuleList([MLP(2, hidden, 2) for _ in range(N - 1)])
        chol = torch.tensor([[1.0, 0.0], [rho, math.sqrt(1 - rho ** 2)]])
        self.register_buffer("chol", chol)

# ---------- 3. Load checkpoint if available ----------
device = torch.device("cpu")
model_path = "deepbsde_sabr.pth"
deep_price_fn = None

if os.path.exists(model_path):
    net = DeepBSDE_SABR().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    @torch.no_grad()
    def deep_price(F0, A0):
        x = torch.tensor([[F0, A0]], dtype=torch.float32, device=device)
        return net.u0_net(x).item()
    deep_price_fn = deep_price
else:
    print(">> Checkpoint not found – analytical curve only will be plotted.")

# ---------- 4. Pricing grids ----------
F_vals = np.linspace(86, 114, 17)  # 60 → 140 step 5
alpha_list = [0.05, 0.10, 0.15]

for alpha in alpha_list:
    model_prices, analytic_prices = [], []
    for F0 in F_vals:
        sigma = hagan_implied_vol(F0, K, T, alpha, beta, rho, nu)
        analytic_prices.append(bs_call(F0, K, T, sigma).item())
        if deep_price_fn:
            model_prices.append(deep_price_fn(F0, alpha))
    # ---- plot ----
    plt.figure()
    if deep_price_fn:
        plt.plot(F_vals, model_prices, label="Deep‑BSDE")
    plt.plot(F_vals, analytic_prices, linestyle="--", label="SABR Analytical")
    plt.title(f"Option Price vs F at α = {alpha}")
    plt.xlabel("Forward F")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.legend()
    plt.show()
