import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

def box_muller_numpy(n: int) -> np.ndarray:
    m = (n + 1) // 2
    u1 = np.random.rand(m)
    u2 = np.random.rand(m)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    z = np.empty(2 * m)
    z[0::2] = z0
    z[1::2] = z1
    return z[:n]

N = 100000
z = box_muller_numpy(N)

k = 5
z_for_chi2 = box_muller_numpy(N * k).reshape(N, k)
chi2_samples = np.sum(z_for_chi2 ** 2, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(z, bins=60, color="skyblue", edgecolor="black", density=True, alpha=0.7)
x = np.linspace(-4, 4, 200)
axes[0].plot(x, norm.pdf(x), "r-", lw=2, label="N(0,1) ")
axes[0].set_title("Normal")
axes[0].set_xlabel("z")
axes[0].set_ylabel("density")
axes[0].legend()
axes[1].hist(chi2_samples, bins=80, color="salmon", edgecolor="black", density=True, alpha=0.7)
x = np.linspace(0, 25, 200)
axes[1].plot(x, chi2.pdf(x, df=k), "k-", lw=2, label=f"χ²(k={k}) ")
axes[1].set_title(f"χ^2")
axes[1].set_xlabel("x")
axes[1].set_ylabel("density")
axes[1].legend()

plt.tight_layout()
plt.show()
