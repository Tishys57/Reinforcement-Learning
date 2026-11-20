import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
import numpy as np
from qlearning import train, evaluate


# Use a consistent style for all plots
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


def moving_average(arr, window):
    """Calculate the moving average over a fixed window."""
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def smooth_runs(data, window):
    """Calculate moving averages for multiple training runs."""
    return np.array([moving_average(run, window) for run in data])


def plot_mean_std(ax, episodes, data, label, window):
    """Plot the mean moving average and standard deviation."""
    smoothed_runs = smooth_runs(data, window)
    mean = np.mean(smoothed_runs, axis=0)
    std = np.std(smoothed_runs, axis=0)

    ax.plot(episodes, mean, label=label)
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)


# Configuration
env = gym.make(
    "FrozenLake-v1", map_name="8x8", is_slippery=True, max_episode_steps=200
)
n_episodes = 50000
alpha = 0.5          # starting learning rate (decays to alpha_end)
alpha_end = 0.01
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay_ratio = 0.8

seeds = [0, 1, 2, 3, 4]
window = 100

algorithms = ["q_learning", "sarsa"]
all_returns = {alg: [] for alg in algorithms}
all_lengths = {alg: [] for alg in algorithms}
greedy_success = {alg: [] for alg in algorithms}

# Save one Q-table to visualize the learned policy later
final_Q_sample = None

# Q-learning vs SARSA across seeds
for alg in algorithms:
    print(f"Training {alg} across {len(seeds)} seeds...")

    for seed in seeds:
        Q_final, returns, lengths, _ = train(
            env,
            n_episodes=n_episodes,
            alpha=alpha,
            alpha_end=alpha_end,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_ratio=eps_decay_ratio,
            algorithm=alg,
            seed=seed,
        )

        all_returns[alg].append(returns)
        all_lengths[alg].append(lengths)
        greedy_success[alg].append(evaluate(env, Q_final, n_episodes=1000))

        if alg == "q_learning" and seed == seeds[0]:
            final_Q_sample = Q_final

for alg in algorithms:
    all_returns[alg] = np.array(all_returns[alg])
    all_lengths[alg] = np.array(all_lengths[alg])


# Success rates
last_1000_returns = all_returns["q_learning"][0][-1000:]
training_success_rate = np.mean(last_1000_returns)
print(f"\nTraining success rate (last 1000 episodes, seed 0): {training_success_rate:.3f}")

if training_success_rate >= 0.7:
    print("✓ Meets the ≥ 0.7 requirement")
else:
    print("✗ Still below 0.7")

print("\nGreedy evaluation (epsilon = 0, 1000 episodes per seed):")
for alg in algorithms:
    rates = np.array(greedy_success[alg])
    print(f"  {alg:10s}: mean = {rates.mean():.3f}, "
          f"per seed = {np.round(rates, 3)}")


# Learning-rate sweep 
alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
sweep_returns = {}

for a in alpha_values:
    print(f"Running Q-learning sweep with alpha = {a}...")
    seed_returns = []

    for seed in seeds:
        _, returns, _, _ = train(
            env,
            n_episodes=n_episodes,
            alpha=a,
            alpha_end=min(alpha_end, a),
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_ratio=eps_decay_ratio,
            algorithm="q_learning",
            seed=seed,
        )
        seed_returns.append(returns)

    sweep_returns[a] = np.array(seed_returns)

env.close()

# Use the same episode range for all smoothed curves
valid_episodes = np.arange(window, n_episodes + 1)


# Figure 1: returns
fig, ax = plt.subplots(figsize=(9, 5))

for label, data in [
    ("Q-learning", all_returns["q_learning"]),
    ("SARSA", all_returns["sarsa"]),
]:
    plot_mean_std(ax, valid_episodes, data, label, window)

ax.set_title(
    "Figure 1: Q-Learning vs. SARSA Returns on FrozenLake-8x8 "
    "(Mean ± Std over 5 Seeds)",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel(f"Return (Moving Avg, window={window})", fontsize=11)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("qlearning_vs_sarsa_returns.png", dpi=300)
plt.show()


# Figure 2: episode lengths
fig, ax = plt.subplots(figsize=(9, 5))

for label, data in [
    ("Q-learning", all_lengths["q_learning"]),
    ("SARSA", all_lengths["sarsa"]),
]:
    plot_mean_std(ax, valid_episodes, data, label, window)

ax.set_title("Figure 2: Episode Lengths During Training", fontsize=12, fontweight="bold")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel(f"Steps per Episode (Moving Avg, window={window})", fontsize=11)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("episode_lengths.png", dpi=300)
plt.show()


# Figure 3: learning-rate sweep
fig, ax = plt.subplots(figsize=(9, 5))

for a, data in sweep_returns.items():
    mean = np.mean(smooth_runs(data, window), axis=0)
    ax.plot(valid_episodes, mean, label=rf"$\alpha_0 = {a}$")

ax.set_title(
    "Figure 3: Q-Learning Learning-Rate Sweep on FrozenLake-8x8",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel(f"Return (Moving Avg, window={window})", fontsize=11)
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("lr_sweep.png", dpi=300)
plt.show()



# Figure 4: epsilon schedule

decay_steps = int(n_episodes * eps_decay_ratio)

epsilons = np.maximum(
    eps_end,
    eps_start - np.arange(n_episodes) / decay_steps * (eps_start - eps_end),
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(epsilons, color="crimson", linewidth=2)

ax.set_title("Figure 4: Linear Epsilon-Decay Schedule", fontsize=12, fontweight="bold")
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel(r"Epsilon ($\epsilon$)", fontsize=11)

fig.tight_layout()
fig.savefig("epsilon_decay.png", dpi=300)
plt.show()

# Figure 5: greedy policy
action_symbols = ["←", "↓", "→", "↑"]
rng_vis = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(7, 7))
ax.matshow(np.zeros((8, 8)), cmap="Pastel1")

# Draw cell borders
ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
ax.tick_params(which="minor", size=0)

for i in range(8):
    for j in range(8):
        state_idx = i * 8 + j
        q_vals = final_Q_sample[state_idx]
        max_q = q_vals.max()

        # Random tie-breaking (same style as training)
        best_act = rng_vis.choice(np.flatnonzero(q_vals == max_q))

        ax.text(
            j, i - 0.15, action_symbols[best_act],
            ha="center", va="center", fontsize=16, fontweight="bold",
        )
        ax.text(
            j, i + 0.25, f"{max_q:.2f}",
            ha="center", va="center", fontsize=8, color="gray",
        )

ax.set_title(
    "Figure 5: Greedy Policy from the Final Q-Table\n"
    "(arrow = best action, number = max Q-value)",
    fontsize=12, fontweight="bold",
)
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
fig.savefig("policy_grid.png", dpi=300)
plt.show()