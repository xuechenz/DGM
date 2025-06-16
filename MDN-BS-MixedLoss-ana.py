import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)
    def forward(self, x):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(x))) + x)

class MDNpdf(nn.Module):
    def __init__(self, K=10, hidden=64, n_layers=5, S0=100.0):
        super().__init__()
        self.K, self.S0 = K, S0
        self.inp   = nn.Linear(2, hidden)
        self.blocks= nn.ModuleList([ResBlock(hidden) for _ in range(n_layers)])
        self.out   = nn.Linear(hidden, 3*K)
    def _split(self, raw):
        logits, mu_hat, logsig_hat = torch.split(raw, self.K, -1)
        pi    = torch.softmax(logits, -1)
        mu    = self.S0 * (1 + 0.3*torch.tanh(mu_hat))
        sigma = 0.2*self.S0 * F.softplus(logsig_hat) + 1e-2
        return pi, mu, sigma
    def forward(self, x):
        t,S = x[:,:1], x[:,1:2]
        S_scaled = (S-self.S0)/self.S0
        h = torch.tanh(self.inp(torch.cat([t, S_scaled],1)))
        for blk in self.blocks: h = blk(h)
        pi, mu, sigma = self._split(self.out(h))
        S = S.expand_as(mu)
        comp = torch.exp(-0.5*((S-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        return (pi*comp).sum(-1, keepdim=True)
    def log_pdf(self,x):
        return torch.log(self(x)+1e-12)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "MDN_forward.pth"       
net = MDNpdf().to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

T_final = 1.0
S_min, S_max = 10.0, 180.0
N_T, N_S = 80, 120               

t_vals = torch.linspace(0.01, T_final, N_T)
S_vals = torch.linspace(S_min, S_max, N_S)
Tg, Sg = torch.meshgrid(t_vals, S_vals, indexing="ij")   
grid   = torch.cat([Tg.reshape(-1,1), Sg.reshape(-1,1)], 1).to(DEVICE)

with torch.no_grad():
    pdf = torch.exp(net.log_pdf(grid)).cpu().numpy().reshape(N_T, N_S)

fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Sg.numpy(), Tg.numpy(), pdf,
                       rstride=1, cstride=1, linewidth=0,
                       antialiased=False, cmap="viridis")

ax.set_xlabel("Stock Price  $S$")
ax.set_ylabel("Time  $t$")
ax.set_zlabel("Density  $p(t,S)$")
ax.set_title("Forward Density learned by MDN")
plt.tight_layout()
plt.show()
