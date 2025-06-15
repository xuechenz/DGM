import math, time
import torch, torch.nn as nn, torch.optim as optim

# ---------------- 0  常量 ----------------
F0, A0   = 100.0, 0.30          # 初始 F, α
F_min, F_max   =  85.0, 115.0      # 例如围绕 strike
A_min, A_max   =   0.05, 0.15      # 即 5% ~ 60% vol
beta     = 0.7                  # β
nu, rho  = 0.4, -0.3            # vol-of-vol & 相关
K        = 100.0
T, N     = 1.0, 100
dt, sqrt_dt = T/N, math.sqrt(T/N)
paths    = 4096
hidden   = 64
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 1  通用 MLP ----------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ---------------- 2  Deep-BSDE for SABR ----------------
class DeepBSDE_SABR(nn.Module):
    def __init__(self):
        super().__init__()
        # Y0 & Z0
        self.u0_net = MLP(2, hidden, 1)
        self.z0_net = MLP(2, hidden, 2)          # 输出 (z1,z2)

        # 其余 N-1 步的 Z_net
        self.z_nets = nn.ModuleList([
            MLP(2, hidden, 2) for _ in range(N-1)
        ])

        # 预先算好相关增量的 Cholesky 矩阵
        self.register_buffer(
            "chol", torch.tensor([[1., 0.],
                                  [rho, math.sqrt(1-rho**2)]])
        )

    # 采样相关 Brownian 增量
    def _dW_dZ(self, batch):
        eps = torch.randn(batch, 2, N, device=device)
        inc = torch.einsum("ij,bjn->bin", self.chol, eps) * sqrt_dt
        return inc[:,0], inc[:,1]                 # (batch,N), (batch,N)

    # 前向：输出 (Y_T - payoff) ，供 MSE
    def forward(self, batch):
        dW, dZ = self._dW_dZ(batch)

        F0_batch = F_min + torch.rand(batch, 1, device=device)*(F_max - F_min)
        A0_batch = A_min + torch.rand(batch, 1, device=device)*(A_max - A_min)

        F = F0_batch.squeeze()   # shape (batch,) 
        a = A0_batch.squeeze()

        y  = self.u0_net(torch.cat([F0_batch, A0_batch], dim=1)).squeeze()
        z  = self.z0_net(torch.cat([F0_batch, A0_batch], dim=1))

        # ===== 时间步迭代 =====
        for n in range(N):
            # 累加 BSDE：dY = z1 dW + z2 dZ
            y = y + z[:,0]*dW[:,n] + z[:,1]*dZ[:,n]

            # 演化 F, α
            F = F + a * F.pow(beta) * dW[:,n]
            a = a + nu * a * dZ[:,n]

            # 下一步 Z
            if n < N-1:
                z = self.z_nets[n](torch.stack([F, a],1))

        payoff = torch.relu(F - K)
        return y - payoff                               # (batch,)

# ---------------- 3  训练 ----------------
model = DeepBSDE_SABR().to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 1500
for ep in range(epochs):
    diff  = model(paths)
    loss  = criterion(diff, torch.zeros_like(diff))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if ep % 100 == 0:
        print(f"Epoch {ep:4d}  loss={loss.item():.4e}")

model.eval()   # ← 加上这一行

with torch.no_grad():
    price = model.u0_net(torch.tensor([[101, A0]], device=device)).item()

print(f"\nOption price at t=0 ≈ {price:.4f}")

torch.save(model.state_dict(), "deepbsde_sabr.pth")
