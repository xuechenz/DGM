import time, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim

T, r, sigma = 1.0, 0.02, 0.05
S_Max, S0   = 200.0, 100.0

def sampler(batch, eps, device, use_gbm=True):
    # Inner-Points
    t = torch.rand(batch,1,device=device) * T
    if use_gbm:
        z = torch.randn_like(t)
        S = S0*torch.exp((r-0.5*sigma**2)*t + sigma*torch.sqrt(t)*z)
    else:
        S = torch.rand(batch,1,device=device) * S_Max
    X_in = torch.cat([t, S], 1)                     

    # IC
    S_ic = S0 + eps*torch.randn(batch,1,device=device)
    X_ic = torch.cat([torch.zeros_like(S_ic), S_ic], 1)

    # BC
    t_b = torch.rand(2*batch,1,device=device) * T
    S_b = torch.cat([torch.zeros(batch,1,device=device),
                     torch.full((batch,1), S_Max, device=device)], 0)
    X_b = torch.cat([t_b, S_b], 1)

    # Mass
    S_mass = torch.rand(batch,1,device=device) * S_Max
    return X_in, X_ic, X_b, S_mass

# MDE Net + ResNet
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d,d), nn.Linear(d,d)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self,x):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(x)))+x)

class MDNpdf(nn.Module):
    def __init__(self, K=10, hidden=64, n_layers=5):
        super().__init__()
        self.K, self.S0 = K, S0
        self.inp   = nn.Linear(2, hidden)
        self.blocks= nn.ModuleList([ResBlock(hidden) for _ in range(n_layers)])
        self.out   = nn.Linear(hidden, 3*K)

    def _split(self, raw):
        logits, mu_hat, logsig_hat = torch.split(raw, self.K, dim=-1)
        pi    = torch.softmax(logits, -1)
        mu    = self.S0*(1+0.3*torch.tanh(mu_hat))
        sigma = 0.2*self.S0*F.softplus(logsig_hat) + 1e-2
        return pi, mu, sigma

    def forward(self, x):                      
        t,S = x[:,:1], x[:,1:2]
        S_scaled = (S-self.S0)/self.S0
        h = torch.tanh(self.inp(torch.cat([t,S_scaled],1)))
        for blk in self.blocks: h = blk(h)
        pi,mu,sigma = self._split(self.out(h))
        S = S.expand_as(mu)
        comp = torch.exp(-0.5*((S-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        return (pi*comp).sum(-1,keepdim=True)

    def log_pdf(self,x):
        return torch.log(self(x)+1e-12)

# PDE Loss
def fp_residual(net, X):
    X.requires_grad_(True)
    p    = net(X)
    grad = torch.autograd.grad(p, X, torch.ones_like(p), create_graph=True)[0]
    p_t, p_S = grad[:,0:1], grad[:,1:2]
    p_SS = torch.autograd.grad(p_S, X, torch.ones_like(p_S), create_graph=True)[0][:,1:2]
    S = X[:,1:2]
    return p_t - ((sigma**2 - r)*p + (2*sigma**2*S - r*S)*p_S + 0.5*sigma**2*S**2*p_SS)

# PDE+IC+BC+Mass+MLE
def loss_fn(net, X_in, X_ic, X_b, S_mass, eps,
            lam=dict(mle=1.0,pde=1.0,ic=1.0,bc=1.0,mass=10.0)):
    # PDE
    loss_pde = (fp_residual(net, X_in)**2).mean()

    # IC
    p_pred = net(X_ic)
    p_true = torch.exp(-(X_ic[:,1:2]-S0)**2/(2*eps**2))/(eps*math.sqrt(2*math.pi))
    loss_ic = ((p_pred-p_true)**2).mean()

    # BC
    loss_bc = (net(X_b)**2).mean()

    # Mass
    t_mass = torch.rand_like(S_mass)*T
    X_mass = torch.cat([t_mass, S_mass], 1)
    loss_mass = ((net(X_mass).mean()*S_Max - 1.0)**2)

    # MLE
    loss_mle = (-net.log_pdf(X_in)).mean()

    total = (lam['pde']*loss_pde + lam['ic']*loss_ic + lam['bc']*loss_bc +
             lam['mass']*loss_mass + lam['mle']*loss_mle)
    return total, loss_pde, loss_ic, loss_bc, loss_mass, loss_mle

# train
if __name__ == "__main__":
    batch, epochs, lr = 2048, 20000, 2e-4
    eps0, eps_min = 0.5, 0.05
    log_every = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MDNpdf(hidden=64).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    start = time.time()
    for ep in range(1, epochs+1):
        eps = max(eps_min, eps0*(0.9995**ep))
        X_in,X_ic,X_b,S_mass = sampler(batch, eps, device)

        loss_tot, loss_pde, loss_ic, loss_bc, loss_mass, loss_mle = \
            loss_fn(net, X_in, X_ic, X_b, S_mass, eps)

        optimizer.zero_grad()
        loss_tot.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        if ep % log_every == 0:
            print(f"Ep {ep:6d} | total={loss_tot.item():.3e} | "
                  f"PDE={loss_pde.item():.2e} | IC={loss_ic.item():.2e} | "
                  f"BC={loss_bc.item():.2e} | Mass={loss_mass.item():.2e} | "
                  f"MLE={loss_mle.item():.2e}")

    print(f"Training finished in {time.time()-start:.1f}s")
    torch.save(net.state_dict(), "MDN_forward.pth")
