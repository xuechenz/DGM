import math, pathlib
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_vol(k, f, t, alpha, beta, rho, volvol):
    f_av = np.sqrt(f * k)
    A = -beta*(2-beta)*alpha**2 / (24 * f_av**(2-2*beta))
    B =  rho*alpha*volvol*beta / (4 * f_av**(1-beta))
    C = (2 - 3*rho**2)*volvol**2 / 24
    FMKR = _f_minus_k_ratio(f, k, beta)
    ZXZ  = _zeta_over_x_of_zeta(k, f, t, alpha, beta, rho, volvol)
    return alpha * FMKR * ZXZ * (1 + (A+B+C)*t)

def _f_minus_k_ratio(f, k, beta):
    eps = 1e-7
    if abs(f-k) > eps:
        if abs(1-beta) > eps:
            return (1-beta)*(f-k)/(f**(1-beta)-k**(1-beta))
        else:
            return (f-k)/np.log(f/k)
    return k**beta

def _zeta_over_x_of_zeta(k, f, t, alpha, beta, rho, volvol):
    eps = 1e-7
    f_av = np.sqrt(f*k)
    zeta = volvol*(f-k)/(alpha*f_av**beta)
    if abs(zeta) > eps:
        return zeta / _x(rho, zeta)
    return 1.0

def _x(rho, z):
    a = (1-2*rho*z+z*z)**0.5 + z - rho
    return np.log(a/(1-rho))

def normal_call(k, f, t, v, r=0.0, cp='call'):
    d = (f-k)/(v*np.sqrt(t))
    s = 1.0 if cp=='call' else -1.0
    return np.exp(-r*t)*(s*(f-k)*norm.cdf(s*d) +
                         v*np.sqrt(t/(2*np.pi))*np.exp(-0.5*d*d))
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DGMNet().to(DEVICE)
net.load_state_dict(torch.load("SABR_DGM_universal.pth", map_location=DEVICE))
net.eval()

beta_vals  = [0.5]
nu_vals    = [0.2]
rho_vals   = [-0.5, 0.0]
alpha_vals = [0.05]
K_values   = [95.0, 100.0]

F_vals   = np.linspace(60, 140, 200, dtype=np.float32)
F_tensor = torch.tensor(F_vals, device=DEVICE).unsqueeze(1)

T_total  = 1.0
t_list   = np.linspace(0.0, T_total, 5, endpoint=False, dtype=np.float32)

out_dir  = pathlib.Path("figs_SABR_normal"); out_dir.mkdir(exist_ok=True)
metrics  = []

with torch.no_grad():
    for beta in beta_vals:
        for nu in nu_vals:
            for rho in rho_vals:
                for t_fixed in t_list:
                    tau = T_total - t_fixed
                    t_tensor = torch.full_like(F_tensor, t_fixed)

                    for alpha in alpha_vals:
                        for K in K_values:
                            X = torch.cat([
                                t_tensor, F_tensor,
                                torch.full_like(F_tensor, alpha),
                                torch.full_like(F_tensor, K),
                                torch.full_like(F_tensor, beta),
                                torch.full_like(F_tensor, nu),
                                torch.full_like(F_tensor, rho)
                            ], dim=1).float()

                            u_pred = net(X).cpu().numpy().ravel()

                            u_exact = np.array([
                                normal_call(
                                    K, float(f), tau,
                                    normal_vol(K, float(f), tau,
                                               alpha, beta, rho, nu)
                                )
                                for f in F_vals
                            ], dtype=np.float32)

                            plt.figure(figsize=(6,4))
                            plt.plot(F_vals, u_pred, label="DGM")
                            plt.plot(F_vals, u_exact, "--", label="Hagan Normal")
                            plt.title(f"β={beta}, ν={nu}, ρ={rho}, t={t_fixed:.2f}, "
                                      f"α={alpha}, K={K}")
                            plt.xlabel("F")
                            plt.ylabel("Price (normal)")
                            plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
                            plt.show()

                            err = np.abs(u_pred - u_exact)
                            rel = err[u_exact>1e-1]/u_exact[u_exact>1e-1]
                            metrics.append({
                                'beta':beta,'nu':nu,'rho':rho,'t':t_fixed,
                                'alpha':alpha,'K':K,
                                'AvgAbs':err.mean(),'MaxAbs':err.max(),
                                'AvgPct':rel.mean()*100
                            })

df = pd.DataFrame(metrics).round({'AvgAbs':6,'MaxAbs':6,'AvgPct':4})
df.set_index(['beta','nu','rho','t','alpha','K'], inplace=True)
print("DGM vs Hagan-Normal")
print(df)
