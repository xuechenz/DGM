import math, itertools, pathlib
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def hagan_iv(F, K, T, alpha, beta, rho, nu,
             eps=1e-12, sigma_cap=3.0):
    if abs(F - K) < eps:    
        term1 = alpha / (F ** (1 - beta))
        term2 = (((1 - beta) ** 2) / 24) * alpha ** 2 / F ** (2 - 2 * beta)
        term2 += (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term2 += ((2 - 3 * rho ** 2) * nu ** 2) / 24
        sigma = term1 * (1 + term2 * T)
        return min(sigma, sigma_cap)

    FK     = F * K
    logFK  = math.log(F / K)
    z      = (nu / alpha) * FK ** ((1 - beta) / 2) * logFK
    x_z_n  = math.sqrt(1 - 2 * rho * z + z * z) + z - rho
    x_z_d  = 1 - rho
    x_z    = math.log((x_z_n + eps) / (x_z_d + eps))
    z      = z + eps

    A = alpha / (
        FK ** ((1 - beta) / 2)
        * (1 + ((1 - beta) ** 2 / 24) * logFK ** 2
           + ((1 - beta) ** 4 / 1920) * logFK ** 4)
    )
    B = z / x_z
    C = 1 + (
        ((1 - beta) ** 2 / 24) * (alpha ** 2) / (FK ** (1 - beta))
        + (rho * beta * nu * alpha) / (4 * FK ** ((1 - beta) / 2))
        + ((2 - 3 * rho ** 2) * nu ** 2) / 24
    ) * T
    sigma = A * B * C
    return min(sigma, sigma_cap)

def bs_call(F, K, T, sigma):
    if sigma < 1e-12:
        return max(F - K, 0.0)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DGMNet().to(DEVICE)
net.load_state_dict(torch.load("SABR_DGM_universal.pth", map_location=DEVICE))
net.eval()

beta_vals = [0.5, 0.7, 0.9]
nu_vals   = [0.2, 0.4]
rho_vals  = [-0.5, 0.0, 0.5]
alpha_list= [0.05, 0.10, 0.15]
K_values  = [95.0, 100.0, 105.0]

F_vals = np.linspace(60, 140, 200, dtype=np.float32)
F_tensor = torch.tensor(F_vals, device=DEVICE).unsqueeze(1)

T = 1.0
t_list = np.linspace(0.0, T, 5, endpoint=False, dtype=np.float32)

metrics = []
out_dir = pathlib.Path("figs_SABR_universal"); out_dir.mkdir(exist_ok=True)

with torch.no_grad():
    for beta in beta_vals:
        for nu in nu_vals:
            for rho in rho_vals:
                for t_fixed in t_list:
                    tau = T - t_fixed
                    t_tensor = torch.full_like(F_tensor, t_fixed)
                    for alpha in alpha_list:
                        for K in K_values:
                            alpha_tensor = torch.full_like(F_tensor, alpha)
                            K_tensor     = torch.full_like(F_tensor, K)
                            beta_tensor  = torch.full_like(F_tensor, beta)
                            nu_tensor    = torch.full_like(F_tensor, nu)
                            rho_tensor   = torch.full_like(F_tensor, rho)
                            X = torch.cat([
                                t_tensor, F_tensor, alpha_tensor,
                                K_tensor, beta_tensor, nu_tensor, rho_tensor
                            ], dim=1).float()

                            u_pred = net(X).cpu().numpy().ravel()

                            u_exact = np.array([
                                bs_call(
                                    float(f), float(K), float(tau),
                                    hagan_iv(
                                        float(f), float(K), float(tau),
                                        alpha, beta, rho, nu
                                    )
                                )
                                for f in F_vals
                            ], dtype=np.float32)

                            plt.figure(figsize=(6,4))
                            plt.plot(F_vals, u_pred, label="DGM")
                            plt.plot(F_vals, u_exact, "--", label="Hagan+BS")
                            plt.title(
                                f"beta={beta}, nu={nu}, rho={rho}, "
                                f"t={t_fixed:.2f}, alpha={alpha:.2f}, K={K}"
                            )
                            plt.xlabel("F")
                            plt.ylabel("Price")
                            plt.legend()
                            plt.grid(alpha=.3)
                            plt.tight_layout()
                            fname = (
                                f"beta{beta}_nu{nu}_rho{rho}_"
                                f"t{t_fixed:.2f}_a{alpha:.2f}_K{K}.png"
                            )
                            plt.savefig(out_dir/fname)
                            plt.close()

                            err = np.abs(u_pred - u_exact)
                            mask = u_exact > 1e-2
                            rel  = err[mask] / u_exact[mask]
                            metrics.append({
                                'beta': beta, 'nu': nu, 'rho': rho,
                                't': float(t_fixed), 'alpha': alpha, 'K': K,
                                'AvgAbs': float(err.mean()),
                                'MaxAbs': float(err.max()),
                                'AvgPct': float(rel.mean()*100)
                            })

df = pd.DataFrame(metrics)
df = df.set_index(['beta','nu','rho','t','alpha','K']) \
       .round({'AvgAbs':6,'MaxAbs':6,'AvgPct':4})
print("\n===== Universal SABR DGM vs Hagan Metrics =====")
print(df)
