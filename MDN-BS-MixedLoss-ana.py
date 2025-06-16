import os, math, torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

T_final = 1.0
S0      = 100.0
y_min, y_max = -4.0, math.log(200/100)   
N_T, N_S = 80, 120                      
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
class DGMNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=5):
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

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "DGM_forward_log.pth"

net = DGMNet().to(DEVICE)
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("请把训练好的 DGM_forward_log.pth 放到当前目录")
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)); net.eval()

t_vals = torch.linspace(0.01, T_final, N_T, device=DEVICE)     
y_vals = torch.linspace(y_min, y_max, N_S, device=DEVICE)
Tg, Yg = torch.meshgrid(t_vals, y_vals, indexing="ij")
grid   = torch.cat([Tg.reshape(-1,1), Yg.reshape(-1,1)], 1)

with torch.no_grad():
    q  = net(grid).cpu().numpy().reshape(N_T, N_S)             
Sg = S0 * torch.exp(Yg).cpu().numpy()
p  = q / Sg                                                    

cap = np.quantile(p, 0.995)
p_clip = np.clip(p, 0, cap)

fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(Sg, Tg.cpu(), p_clip,
                       rstride=1, cstride=1,
                       cmap="viridis", linewidth=0, antialiased=True)
ax.set_xlabel("Stock Price  $S$")
ax.set_ylabel("Time  $t$")
ax.set_zlim3d(0, 0.3)
ax.set_zlabel("Density  $p(t,S)$")
ax.set_title("Forward Density learned by DGM")
ax.view_init(elev=30, azim=-135)
fig.colorbar(surf, shrink=0.6)
plt.tight_layout()
plt.show()
