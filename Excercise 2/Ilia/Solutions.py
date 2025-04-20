import numpy as np
import matplotlib.pyplot as plt

# Exercise 1: Generate Lorenz-63 reference trajectory
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
h = 0.01
t_final = 200.0
n_steps = int(t_final / h)
tout = 0.05
L = int(tout / h)  # steps between outputs
n_out = n_steps // L + 1

# Pre-allocate storage
trajectory = np.zeros((3, n_out))
z = np.array([1.0, 0.0, 0.0])

trajectory[:, 0] = z
out_idx = 1

for i in range(1, n_steps + 1):
    x, y, zc = z
    dz = np.array([
        sigma * (y - x),
        x * (rho - zc) - y,
        x * y - beta * zc
    ])
    z = z + h * dz
    if i % L == 0:
        trajectory[:, out_idx] = z
        out_idx += 1

# Plot 3D Lorenz attractor (sampled)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(projection='3d')
ax.plot(trajectory[0], trajectory[1], trajectory[2], lw=0.5)
ax.set_title("Lorenz-63 Reference Trajectory")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout()
plt.show()

# Exercise 2: Generate observations
np.random.seed(0)  # for reproducibility
n_obs = n_out - 1  # observations start from t = 0.05 to t=200
xref = trajectory[0, 1:]
obs = np.zeros(n_obs)
for k in range(n_obs):
    noise_samples = np.random.randn(20)
    obs[k] = xref[k] + (1/20) * noise_samples.sum()

# Plot a short segment of observations and error
t_obs = np.linspace(tout, t_final, n_obs)
plt.figure(figsize=(6, 3))
plt.plot(t_obs[:400], xref[:400], label="True x")
plt.plot(t_obs[:400], obs[:400], '.', markersize=2, label="Observed x")
plt.legend()
plt.title("True vs Observed (first 400 samples)")
plt.tight_layout()
plt.show()

# Exercise 3: Linear extrapolation forecasts
# 1-step ahead (Δtout = 0.05)
y = obs
pred1 = np.empty_like(y)
pred1[:1] = np.nan
pred1[1:] = y[0:-1] + (y[0:-1] - np.concatenate(([y[0]], y[:-2])))
# Align predictions and compare pred1[1:] to y[1:]
rmse1 = np.sqrt(np.mean((y[1:] - pred1[1:])**2))

# 3-step ahead (Δtout = 0.15)
pred3 = np.empty_like(y)
pred3[:1] = np.nan
pred3[1:] = y[0:-1] + 3 * (y[0:-1] - np.concatenate(([y[0]], y[:-2])))
# Align predictions and compare pred3[2:] to y[2:]
rmse3 = np.sqrt(np.mean((y[2:] - pred3[2:])**2))

print(f"Time-averaged RMSE for Δtout = 0.05: {rmse1:.4f}")
print(f"Time-averaged RMSE for Δtout = 0.15: {rmse3:.4f}")

# Plot sample of actual vs 1-step ahead forecast
plt.figure(figsize=(6, 3))
plt.plot(t_obs[2:300], y[2:300], label="Observed x")
plt.plot(t_obs[2:300], pred1[2:300], label="1-step forecast")
plt.legend()
plt.title("Observed vs 1-step Forecast (samples 2–300)")
plt.tight_layout()
plt.show()

# Plot sample of actual vs 3-step ahead forecast
plt.figure(figsize=(6, 3))
plt.plot(t_obs[2:300], y[2:300], label="Observed x")
plt.plot(t_obs[2:300], pred3[2:300], label="3-step forecast")
plt.legend()
plt.title("Observed vs 3-step Forecast (samples 2–300)")
plt.tight_layout()
plt.show()