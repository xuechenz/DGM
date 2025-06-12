import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

class DGMBlock(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.U_z, self.W_z = torch.nn.Linear(in_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_g, self.W_g = torch.nn.Linear(in_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_r, self.W_r = torch.nn.Linear(in_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_h, self.W_h = torch.nn.Linear(in_dim, hidden_dim), torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = torch.nn.Tanh()

    def forward(self, x, s_prev, s_first):
        z = self.act(self.U_z(x) + self.W_z(s_prev))
        g = self.act(self.U_g(x) + self.W_g(s_first))
        r = self.act(self.U_r(x) + self.W_r(s_prev))
        h = self.act(self.U_h(x) + self.W_h(s_prev * r))
        return (1.0 - g) * h + z * s_prev


class DGMNet(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, n_layers=3):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList([DGMBlock(input_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = torch.nn.Linear(hidden_dim, 1)
        self.act = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        s = self.act(self.input_layer(x))
        s_first = s
        for blk in self.blocks:
            s = blk(x, s, s_first)
        return self.softplus(self.output_layer(s))


net = DGMNet().cpu()
net.load_state_dict(torch.load("DGM_forward.pth", map_location="cpu"))
net.eval()

T, S_max = 1.0, 200.0
t_vals  = np.linspace(0.0, T,   60)            
S_vals  = np.linspace(0.0, S_max, 120)        
T_grid, S_grid = np.meshgrid(t_vals, S_vals)
grid = np.stack([T_grid.ravel(), S_grid.ravel()], axis=1)
with torch.no_grad():
    dens = net(torch.tensor(grid, dtype=torch.float32)).numpy().reshape(T_grid.shape)

fig = plt.figure(figsize=(8, 5))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, T_grid, dens, linewidth=0, antialiased=False)
ax.set_xlabel("Stock Price $S$")
ax.set_ylabel("Time $t$")
ax.set_zlabel("Density $p(t,S)$")
ax.set_title("Learned Forward Density $p(t,S)$")
plt.tight_layout()
plt.show()
