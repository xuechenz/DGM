import torch, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn as nn

T, K, SIGMA, S_MAX = 1.0, 100.0, 0.05, 200.0        
r_values = [0.05, 0.07]                            
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Tanh()
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + identity)

class ResNetDGM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)

net = ResNetDGM(hidden_dim=32).to(device)
ckpt_path = "DGM_0.001_2048_10000.pth"      
net.load_state_dict(torch.load(ckpt_path, map_location=device))
net.eval()

time_points = [0.15, 0.45, 0.75, 0.90]
S = torch.linspace(60, 140, 120, device=device).unsqueeze(1)
S_np = S.cpu().numpy().flatten()

for r_cur in r_values:
    plt.figure(figsize=(8, 5))
    for t in time_points:
        tau = T - t
        d1 = (np.log(S_np / K) + (r_cur + 0.5 * SIGMA**2) * tau) / (SIGMA * math.sqrt(tau))
        delta_exact = norm.cdf(d1)

        t_tensor = torch.full_like(S, t)
        r_tensor = torch.full_like(S, r_cur)        
        X = torch.cat([t_tensor, S, r_tensor], dim=1).requires_grad_(True)
        u = net(X)
        grad = torch.autograd.grad(u, X,
                                   grad_outputs=torch.ones_like(u),
                                   create_graph=False)[0][:, 1]
        delta_pred = grad.detach().cpu().numpy().flatten()

        plt.plot(S_np, delta_pred,  label=f'Pred t={t:.2f}')
        plt.plot(S_np, delta_exact, '--', alpha=0.7)

    plt.title(f'Delta vs S  (r = {r_cur:.2%})')
    plt.xlabel('Stock Price S')
    plt.ylabel('Delta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
