import torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      

device = torch.device("cpu")
net = DGMNet(input_dim=2, hidden_dim=128, n_layers=4).to(device)
net.load_state_dict(torch.load("DGM_pde_ic_ref.pth", map_location=device))
net.eval()

T, S0, S_max = 1.0, 100.0, 200.0
t_vals = np.linspace(0.0, T,  60)   
S_vals = np.linspace(0.0, S_max, 120)
T_grid, S_grid = np.meshgrid(t_vals, S_vals, indexing='ij')
grid = np.stack([T_grid.ravel(), S_grid.ravel()], axis=1).astype(np.float32)

with torch.no_grad():
    x = torch.from_numpy(grid).to(device)
    dens = net(x).cpu().numpy().reshape(T_grid.shape)

fig = plt.figure(figsize=(10, 6))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    S_grid, T_grid, dens,
    rstride=2, cstride=2,
    cmap='viridis', linewidth=0, antialiased=True
)
ax.set_xlabel("Stock Price $S$")
ax.set_ylabel("Time $t$")
ax.set_zlabel("Density $p(t,S)$")
ax.set_title("DGM-Learned Forward Density")
ax.view_init(elev=30, azim=-60)          
fig.colorbar(surf, shrink=0.5, aspect=10, label='Density')
plt.tight_layout()
plt.show()
