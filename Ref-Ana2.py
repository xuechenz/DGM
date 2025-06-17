import torch, numpy as np, matplotlib.pyplot as plt

device = next(net.parameters()).device          
S0, r, sigma = 100.0, 0.02, 0.05

def bs_forward_density(t, S, S0=100., r=0.02, sigma=0.05):
    S = np.maximum(S, 1e-8)
    var  = sigma**2 * t
    coef = 1/(S * np.sqrt(2*np.pi*var))
    expo = - (np.log(S/S0)-(r-0.5*sigma**2)*t)**2/(2*var)
    return coef * np.exp(expo)

time_list = [0.1, 0.2, 0.5, 0.8]

for t_plot in time_list:
    S_coarse = np.linspace(1, 200, 300)

    width = 5 * sigma * np.sqrt(t_plot)
    zone  = (S0 - width, S0 + width)
    S_fine = np.linspace(zone[0], zone[1], 2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, S_plot, subtitle in [
        (ax1, S_coarse,  "Global view"),
        (ax2, S_fine,    f"Zoom around $S_0\\pm{width:.3f}$")
    ]:
        with torch.no_grad():
            t_tensor = torch.full((len(S_plot), 1), t_plot, dtype=torch.float32, device=device)
            S_tensor = torch.tensor(S_plot, dtype=torch.float32, device=device).unsqueeze(1)
            p_pred   = net(torch.cat([t_tensor, S_tensor], dim=1)).cpu().numpy().squeeze()

        p_true = bs_forward_density(t_plot, S_plot, S0=S0, r=r, sigma=sigma)

        ax.plot(S_plot, p_pred, label="DGM pred", lw=2)
        ax.plot(S_plot, p_true, '--', label="Analytic", lw=2)
        ax.set_title(f"$t={t_plot}$ — {subtitle}")
        ax.set_xlabel("Stock Price $S$")
        ax.set_ylabel("$p(t,S)$")
        ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.show()
