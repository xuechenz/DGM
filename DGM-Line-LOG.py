import os, math, torch, numpy as np, matplotlib.pyplot as plt

S0, r, sigma = 100., 0.02, 0.05
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = "DGM_forward_log.pth"       

net = DGMNet().to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

def bs_forward_density(t, S):
    S   = np.maximum(S, 1e-8)
    var = sigma**2 * t
    coef= 1/(S*np.sqrt(2*np.pi*var))
    expo= - (np.log(S/S0)-(r-0.5*sigma**2)*t)**2 / (2*var)
    return coef*np.exp(expo)

time_list = [0.1, 0.2, 0.5, 0.8]
for t_plot in time_list:
    S_coarse = np.linspace(1, 200, 300)
    width    = 5 * sigma * np.sqrt(t_plot)
    zone     = (S0 - width, S0 + width)
    S_fine   = np.linspace(zone[0], zone[1], 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, S_vec, subtitle in [
        (ax1, S_coarse, "Global view"),
        (ax2, S_fine,   f"Zoom around $S_0\\!\\pm\\!{width:.3f}$")
    ]:
        S_tensor = torch.tensor(S_vec, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        t_tensor = torch.full_like(S_tensor, t_plot)
        y_tensor = torch.log(S_tensor / S0)
        with torch.no_grad():
            q_pred = net(torch.cat([t_tensor, y_tensor], 1)).cpu().numpy().squeeze()
        p_pred = q_pred / S_vec                    

        p_true = bs_forward_density(t_plot, S_vec)

        ax.plot(S_vec, p_pred, label="DGM pred", lw=2)
        ax.plot(S_vec, p_true, '--', label="Analytic", lw=2)
        ax.set_title(f"$t={t_plot}$ — {subtitle}")
        ax.set_xlabel("$S$"); ax.set_ylabel("$p(t,S)$")
        ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()
