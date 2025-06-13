import math, time
import torch, torch.nn as nn, torch.optim as optim

F0, A0   = 100.0, 0.3         
beta     = 0.7
nu, rho  = 0.4, -0.3           
T, N     = 1.0, 100
dt       = T / N
sqrt_dt  = math.sqrt(dt)
K        = 100.0            
path_num = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1):
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
        return self.fc3(x)

def make_z_net():
    return MLP(in_dim=2, hidden_dim=64, out_dim=2).to(device)  

class DeepBSDE_SABR(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.Y0 = nn.Parameter(torch.tensor([0.0], device=device))
        self.z_nets = nn.ModuleList([make_z_net() for _ in range(N)])

    def forward(self, dW, dZ):
        batch = dW.size(0)
        F = torch.full((batch,1), F0, device=device)
        a = torch.full((batch,1), A0, device=device)
        Y = self.Y0.expand(batch, 1)

        for n in range(N):
            z_hat = self.z_nets[n](torch.cat([F, a], dim=-1))  
            Z1 = a * F.pow(beta) * z_hat[:, :1]                 
            Z2 = nu * a       * z_hat[:, 1:]                 
            Y  = Y + Z1 * dW[:, n, :] + Z2 * dZ[:, n, :]

            F = F + a * F.pow(beta) * dW[:, n, :]
            a = a + nu * a * dZ[:, n, :]

        payoff = torch.relu(F - K)
        return (Y - payoff).squeeze()           

model = DeepBSDE_SABR(hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
epochs = 2000
for epoch in range(epochs):
    Z1 = torch.randn(path_num, N, 1, device=device)
    Z2 = torch.randn(path_num, N, 1, device=device)
    dW = sqrt_dt * Z1
    dZ = sqrt_dt * (rho * Z1 + math.sqrt(1 - rho ** 2) * Z2)

    optimizer.zero_grad()
    diff  = model(dW, dZ)                      
    loss  = criterion(diff, torch.zeros_like(diff))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss={loss.item():.3e}")

model_path = f"SABR-BSDE-{1e-3:.0e}-{epochs}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model Saved: {model_path}")
