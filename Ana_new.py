# ========== 0. 依赖 ==========
import math, itertools, pathlib
import torch, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

FILE   = "DGM_0.001_2048_10000.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 1. 载入网络 ==========
net = DGMNet(input_dim=5, hidden_dim=32, n_layers=3).to(DEVICE)
net.load_state_dict(torch.load(FILE, map_location=DEVICE))
net.eval()

# ========== 2. 参数 ==========
T          = 1.0
t_fixed    = 0.0
r_values   = [0.02, 0.05, 0.08]
sig_values = [0.05, 0.06, 0.09]
K_values   = [95.0, 100.0, 105.0]             # ← 行权价列表
S_tensor   = torch.linspace(60, 140, 200, device=DEVICE).unsqueeze(1)
S_np       = S_tensor.cpu().numpy().ravel()

# ========== 3. 画图 ==========
out_dir = pathlib.Path("figs_K"); out_dir.mkdir(exist_ok=True)

for r_cur, sig_cur, K_cur in itertools.product(r_values, sig_values, K_values):
    # 组装 (t,S,r,σ,K)
    t_tensor  = torch.full_like(S_tensor, t_fixed)
    r_tensor  = torch.full_like(S_tensor, r_cur)
    sig_tensor= torch.full_like(S_tensor, sig_cur)
    K_tensor  = torch.full_like(S_tensor, K_cur)
    X = torch.cat([t_tensor, S_tensor, r_tensor, sig_tensor, K_tensor], dim=1)

    # 网络预测
    with torch.no_grad():
        u_pred = net(X).cpu().numpy().ravel()

    # Black-Scholes 解析价
    d1 = (np.log(S_np / K_cur) + (r_cur + 0.5*sig_cur**2)*T) / (sig_cur*math.sqrt(T))
    d2 = d1 - sig_cur*math.sqrt(T)
    u_exact = S_np*norm.cdf(d1) - K_cur*np.exp(-r_cur*T)*norm.cdf(d2)

    # 绘图
    plt.figure(figsize=(6.4,4.5))
    plt.plot(S_np, u_pred, label="DGM")
    plt.plot(S_np, u_exact, "--", label="Black-Scholes")
    plt.title(f"t=0 | r={r_cur:.2%}, σ={sig_cur:.2%}, K={K_cur}")
    plt.xlabel("Stock Price $S$")
    plt.ylabel("Call Price $u(0,S)$")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.show()


# ========== 4. 误差表 ==========
metrics = []
with torch.no_grad():
    for r_cur, sig_cur, K_cur in itertools.product(r_values, sig_values, K_values):
        t_tensor  = torch.zeros_like(S_tensor)
        r_tensor  = torch.full_like(S_tensor, r_cur)
        sig_tensor= torch.full_like(S_tensor, sig_cur)
        K_tensor  = torch.full_like(S_tensor, K_cur)
        X = torch.cat([t_tensor, S_tensor, r_tensor, sig_tensor, K_tensor], dim=1)
        u_pred = net(X).cpu().numpy().ravel()

        d1 = (np.log(S_np / K_cur) + (r_cur + 0.5*sig_cur**2)*T) / (sig_cur*math.sqrt(T))
        d2 = d1 - sig_cur*math.sqrt(T)
        u_exact = S_np*norm.cdf(d1) - K_cur*np.exp(-r_cur*T)*norm.cdf(d2)

        abs_err = np.abs(u_pred - u_exact)
        mask    = u_exact >= 1.0
        rel_err = abs_err[mask] / u_exact[mask]

        metrics.append(dict(K=K_cur, r=r_cur, σ=sig_cur,
                            AvgAbs=abs_err.mean(),
                            MaxAbs=abs_err.max(),
                            AvgPct=rel_err.mean()*100))

df = (pd.DataFrame(metrics)
        .set_index(["K","r","σ"])
        .round({"AvgAbs":6,"MaxAbs":6,"AvgPct":4}))

print("\n===== 误差统计 (S∈[60,140]) =====")
print(df)
