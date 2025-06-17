import numpy as np
from pathlib import Path

# ===== 基础参数 =====
S0, r, sigma = 100.0, 0.02, 0.05
T            = 1.0
dt           = 1/504          # ⬅ 时间步减半
n_paths      = 50_000         # ⬅ 更多路径
t_list = np.round(np.linspace(0.05, 1.0, 20), 3)   # ⬅ 20 个时间切片

# ===== 价格网格 (更细 bin) =====
S_min, S_max, dS =  0.0, 200.0, 0.2               # ⬅ bin 宽度改为 0.2
S_edges   = np.arange(S_min, S_max + dS, dS)
S_centers = 0.5 * (S_edges[:-1] + S_edges[1:])
bin_width = dS

# ===== Monte-Carlo 采样 =====
def sample_gbm_paths(S0, r, sigma, T, dt, n_paths, file_path, overwrite=False):
    fp = Path(file_path)
    if fp.exists() and not overwrite:
        data = np.load(fp)
        return data["times"], data["paths"]

    n_steps = int(T / dt) + 1
    times   = np.linspace(0, T, n_steps, dtype=np.float32)

    dW = np.random.normal(scale=np.sqrt(dt), size=(n_paths, n_steps - 1)).astype(np.float32)
    W  = np.concatenate([np.zeros((n_paths, 1), dtype=np.float32),
                         np.cumsum(dW, axis=1)], axis=1)

    drift = (r - 0.5 * sigma**2) * times
    paths = S0 * np.exp(drift + sigma * W)

    np.savez_compressed(fp, times=times, paths=paths)
    return times, paths

times, paths = sample_gbm_paths(S0, r, sigma, T, dt, n_paths,
                                "gbm_samples_mc.npz", overwrite=True)

# ===== 生成更细的密度矩阵 =====
density_matrix = []
for t in t_list:
    idx = np.abs(times - t).argmin()    # 找到最近的时间索引
    S_slice = paths[:, idx]
    counts, _ = np.histogram(S_slice, bins=S_edges)
    density_matrix.append(counts / (n_paths * bin_width))

density_matrix = np.array(density_matrix, dtype=np.float32)

np.savez_compressed("gbm_density_hist.npz",
                    t_list=t_list.astype(np.float32),
                    S_centers=S_centers.astype(np.float32),
                    density=density_matrix)

print("density_matrix shape:", density_matrix.shape)
