import torch
import torch.nn as nn
import torch.optim as optim
import time

def sampler(batch_size, T, S_Max, R_Min, R_Max, device):
    # Internal points sample (t, S, r) uniformly in [0,T] * [0,S_MAX] * [r_min, r_max]
    t   = torch.rand(batch_size,1, device=device)*T
    S   = torch.rand(batch_size,1, device=device)*S_Max
    r   = torch.rand(batch_size,1, device=device)*(R_Max-R_Min)+R_Min   
    X_in = torch.cat([t, S, r], dim=1)

    # Terminal points: t = T, S, r in [0, S_MAX, [r_min, r_max]]
    t_T = torch.full_like(t, T)
    S_T = torch.rand_like(S)*S_Max
    r_T = torch.rand_like(r) * (R_Max - R_Min) + R_Min
    X_T = torch.cat([t_T, S_T, r_T], dim=1)

    # Boundary condition points:
    t_b = torch.rand_like(t)
    r_b = torch.rand_like(r) * (R_Max - R_Min) + R_Min
    X_b0 = torch.cat([t_b, torch.zeros_like(S), r_b], dim=1)
    X_b1 = torch.cat([t_b, torch.full_like(S, S_Max), r_b], dim=1)
    X_b  = torch.cat([X_b0, X_b1], dim=0)
    return X_in, X_T, X_b

def pde_residual(net, X, SIGMA):
    # Use autograd find derivatives 
    X = X.clone().requires_grad_(True)
    u = net(X)
    grads = torch.autograd.grad(u, X, torch.ones_like(u),
                                create_graph=True)[0]
    u_t, u_S = grads[:, 0:1], grads[:, 1:2]
    u_SS = torch.autograd.grad(u_S, X, torch.ones_like(u_S),
                               create_graph=True)[0][:, 1:2]
    t, S, r = X[:, 0:1], X[:, 1:2], X[:, 2:3]  
    return u_t + 0.5*SIGMA**2*S**2*u_SS + r*S*u_S - r*u


def loss_fn(net, X_in, X_T, X_b, K, S_Max, T, SIGMA):
    res = pde_residual(net, X_in, SIGMA)
    loss_pde = torch.mean(res**2)

    u_T = net(X_T)
    payoff = torch.relu(X_T[:, 1:2] - K) # Payoff at terminal 
    loss_T = torch.mean((u_T - payoff)**2)

    u_b   = net(X_b) # Boundary prediction
    t_b   = X_b[:, 0:1]
    r_b   = X_b[:, 2:3]
    N_b   = X_b.shape[0] // 2
    bc0_diff = u_b[:N_b]
    bc1_diff = u_b[N_b:] - (S_Max - K*torch.exp(-r_b[N_b:]*(T - t_b[N_b:]))) # Use r calculate loss
    loss_b = torch.mean(bc0_diff**2) + torch.mean(bc1_diff**2)
    total = loss_pde + loss_T + loss_b
    return total, loss_pde, loss_T, loss_b

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
        # residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        # Tanh activation for input layer
        self.act = nn.Tanh()
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)

if __name__ == "__main__":
    T, K, R, SIGMA, S_Max = 1.0, 100.0, 0.02, 0.05, 200.0
    batch, epochs, lr = 2048, 10000, 1e-3
    R_Min, R_Max = 0.0, 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model and optimizer
    net = ResNetDGM(hidden_dim=32).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    start_time = time.time()
    # training loop
    for epoch in range(1, epochs+1):
        X_in, X_T, X_b = sampler(batch, T, S_Max, R_Min, R_Max, device)
        loss, l_pde, l_T, l_b = loss_fn(net, X_in, X_T, X_b, K, S_Max, T, SIGMA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:>4}: total={loss.item():.2e}, PDE={l_pde:.2e}, "
                  f"T={l_T:.2e}, B={l_b:.2e}")
            
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.") 

    filename = f"DGM_{lr}_{batch}_{epochs}.pth"
    torch.save(net.state_dict(), filename)
    print(f"Model saved to {filename}")
