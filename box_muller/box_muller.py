import numpy as np
import matplotlib.pyplot as plt

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
    return z[:n], u1, u2

N = 100000
z, u1, u2 = box_muller_numpy(N)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(u1, bins=50, color="skyblue", edgecolor="black")
axes[0].set_title("Uniform Distribution U(0,1)")
axes[0].set_xlabel("u1")
axes[0].set_ylabel("Frequency")
axes[1].scatter(u1[:2000], u2[:2000], s=5, alpha=0.5, color="green")
axes[1].set_title("Uniform Samples (u1, u2)")
axes[1].set_xlabel("u1")
axes[1].set_ylabel("u2")
axes[2].hist(z, bins=60, color="salmon", edgecolor="black", density=True)
axes[2].set_title("Generated Normal Distribution N(0,1)")
axes[2].set_xlabel("z")
axes[2].set_ylabel("Density")

plt.tight_layout()
plt.show()
