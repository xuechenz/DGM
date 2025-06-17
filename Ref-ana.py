import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
density_file = "gbm_density_hist.npz"
d        = np.load(density_file)        
t_grid   = torch.tensor(d["t_list"],    dtype=torch.float32)
S_grid   = torch.tensor(d["S_centers"], dtype=torch.float32)
pdf      = torch.tensor(d["density"],   dtype=torch.float32)

mask     = (S_grid >= S0-S_window) & (S_grid <= S0+S_window)
TT, SS   = torch.meshgrid(t_grid, S_grid[mask], indexing='ij')
XY_all   = torch.stack([TT.flatten(), SS.flatten()], 1)
v_all    = pdf[:, mask].flatten().unsqueeze(1)

perm     = torch.randperm(XY_all.size(0))[:N_REF]
XY_ref_cpu, v_ref_cpu = XY_all[perm], v_all[perm]
NUM_REF  = XY_ref_cpu.size(0)
def plot_mc_reference_3d(XY_ref_cpu, v_ref_cpu):
    t_vals = XY_ref_cpu[:, 0].numpy()
    S_vals = XY_ref_cpu[:, 1].numpy()
    p_vals = v_ref_cpu[:, 0].numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(S_vals, t_vals, p_vals, c=p_vals, cmap='viridis', s=5)

    ax.set_xlabel("Stock Price $S$")
    ax.set_ylabel("Time $t$")
    ax.set_zlabel("Density $p(t, S)$")
    ax.set_title("Monte Carlo Sampled Density (Reference Data)")
    fig.colorbar(sc, shrink=0.5, aspect=5, label="Density")
    plt.tight_layout()
    plt.show()

plot_mc_reference_3d(XY_ref_cpu, v_ref_cpu)
