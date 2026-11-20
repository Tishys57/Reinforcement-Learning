import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn import train_dqn

torch.set_num_threads(1)
CPU = torch.device("cpu")
SEEDS = [0, 1, 2, 3, 4]


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def first_episode_hitting(ma_values: np.ndarray, threshold: float = 475.0, window: int = 100) -> Optional[int]:
    hits = np.where(ma_values >= threshold)[0]
    if len(hits) == 0:
        return None
    return int(hits[0] + window)


def run_condition(name: str, target_sync_freq: int, num_episodes: int, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    all_returns: List[List[float]] = []
    all_eval_returns: List[List[float]] = []
    all_q_means: List[float] = []
    all_q_maxes: List[float] = []
    all_loss_means: List[float] = []
    seed_hit_episodes: List[Optional[int]] = []

    print(f"\n=== Running {name} (target_sync_freq={target_sync_freq}, episodes={num_episodes}) across seeds {SEEDS} ===", flush=True)
    for seed in SEEDS:
        t0 = time.time()
        print(f"  Starting seed {seed}...", end=" ", flush=True)
        logs = train_dqn(
            seed=seed,
            num_episodes=num_episodes,
            target_sync_freq=target_sync_freq,
            device=CPU,
        )
        ep_ret = np.array(logs["episode_returns"], dtype=np.float64)
        ma100 = moving_average(ep_ret, window=100)
        hit_ep = first_episode_hitting(ma100, threshold=475.0, window=100)
        seed_hit_episodes.append(hit_ep)

        all_returns.append(logs["episode_returns"])
        all_eval_returns.append(logs["eval_returns"])
        all_q_means.append(float(np.mean(logs["mean_max_q"])))
        all_q_maxes.append(float(np.max(logs["mean_max_q"])))
        all_loss_means.append(float(np.mean(logs["losses"])))

        print(
            f"done in {time.time()-t0:.1f}s | Best MA100 = {np.max(ma100):.2f}, "
            f"First MA>=475 ep = {hit_ep}, Final Eval = {logs['eval_returns'][-1]:.1f}, "
            f"Mean/Max Q = {all_q_means[-1]:.1f}/{all_q_maxes[-1]:.1f}",
            flush=True,
        )

    returns_mat = np.array(all_returns, dtype=np.float64)
    mean_curve = returns_mat.mean(axis=0)
    std_curve = returns_mat.std(axis=0)

    ma100_mat = np.array([moving_average(r, 100) for r in returns_mat])
    ma100_mean = ma100_mat.mean(axis=0)
    ma100_std = ma100_mat.std(axis=0)
    mean_hit_ep = first_episode_hitting(ma100_mean, threshold=475.0, window=100)

    eval_mat = np.array(all_eval_returns, dtype=np.float64)
    eval_mean = eval_mat.mean(axis=0)
    eval_std = eval_mat.std(axis=0)

    summary = {
        "name": name,
        "target_sync_freq": target_sync_freq,
        "seeds": SEEDS,
        "num_episodes": num_episodes,
        "returns": all_returns,
        "eval_returns": all_eval_returns,
        "eval_episodes": logs["eval_episodes"],
        "eval_mean": eval_mean.tolist(),
        "eval_std": eval_std.tolist(),
        "seed_hit_episodes_475": seed_hit_episodes,
        "mean_curve_hit_episode_475": mean_hit_ep,
        "best_mean_ma100": float(np.max(ma100_mean)),
        "mean_curve": mean_curve.tolist(),
        "std_curve": std_curve.tolist(),
        "ma100_mean": ma100_mean.tolist(),
        "ma100_std": ma100_std.tolist(),
        "seed_q_means": all_q_means,
        "seed_q_maxes": all_q_maxes,
        "seed_loss_means": all_loss_means,
    }

    with open(os.path.join(out_dir, f"{name}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"-> {name} 5-seed mean 100-ep MA first hits 475 at episode: {mean_hit_ep}", flush=True)
    print(f"-> {name} Best 5-seed mean 100-ep MA: {summary['best_mean_ma100']:.2f}", flush=True)
    return summary


def plot_baseline(baseline: Dict, out_dir: str) -> None:
    episodes = np.arange(1, baseline["num_episodes"] + 1)
    mean = np.array(baseline["mean_curve"])
    std = np.array(baseline["std_curve"])

    ma_episodes = np.arange(100, baseline["num_episodes"] + 1)
    ma_mean = np.array(baseline["ma100_mean"])
    ma_std = np.array(baseline["ma100_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, mean, alpha=0.30, color="tab:blue", label="Raw Return (5-seed mean)")
    ax.fill_between(episodes, np.clip(mean - std, 0, 500), np.clip(mean + std, 0, 500), alpha=0.12, color="tab:blue")

    ax.plot(ma_episodes, ma_mean, color="tab:blue", linewidth=2.2, label="100-ep Moving Avg (mean ± std)")
    ax.fill_between(ma_episodes, ma_mean - ma_std, ma_mean + ma_std, alpha=0.25, color="tab:blue")

    ax.axhline(475, color="tab:green", linestyle="--", linewidth=1.5, label="Solved Threshold (475)")
    hit_ep = baseline["mean_curve_hit_episode_475"]
    if hit_ep is not None:
        ax.axvline(hit_ep, color="tab:red", linestyle=":", linewidth=1.5, label=f"First MA ≥ 475 (Ep {hit_ep})")

    ax.set_title("DQN Baseline Learning Curve on CartPole-v1 (5 Seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_ylim(0, 530)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_learning_curve.png"), dpi=300)
    plt.close(fig)


def plot_comparison(baseline: Dict, no_target: Dict, out_dir: str) -> None:
    ma_episodes = np.arange(100, baseline["num_episodes"] + 1)
    b_mean, b_std = np.array(baseline["ma100_mean"]), np.array(baseline["ma100_std"])
    n_mean, n_std = np.array(no_target["ma100_mean"]), np.array(no_target["ma100_std"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ma_episodes, b_mean, color="tab:blue", linewidth=2.2, label="Baseline (sync every 500 steps)")
    ax1.fill_between(ma_episodes, b_mean - b_std, b_mean + b_std, alpha=0.2, color="tab:blue")
    ax1.plot(ma_episodes, n_mean, color="tab:orange", linewidth=2.2, label="No Target Network (sync every step)")
    ax1.fill_between(ma_episodes, n_mean - n_std, n_mean + n_std, alpha=0.2, color="tab:orange")
    ax1.axhline(475, color="tab:green", linestyle="--", linewidth=1.3, label="Solved Threshold (475)")
    ax1.set_title("100-Episode Moving Average Return")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return (mean ± std)")
    ax1.set_ylim(0, 530)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")

    ev_eps = np.array(baseline["eval_episodes"])
    be_mean, be_std = np.array(baseline["eval_mean"]), np.array(baseline["eval_std"])
    ne_mean, ne_std = np.array(no_target["eval_mean"]), np.array(no_target["eval_std"])

    ax2.plot(ev_eps, be_mean, marker="o", markersize=4, color="tab:blue", linewidth=2.0, label="Baseline Eval (ε=0)")
    ax2.fill_between(ev_eps, np.clip(be_mean - be_std, 0, 500), np.clip(be_mean + be_std, 0, 500), alpha=0.2, color="tab:blue")
    ax2.plot(ev_eps, ne_mean, marker="s", markersize=4, color="tab:orange", linewidth=2.0, label="No Target Eval (ε=0)")
    ax2.fill_between(ev_eps, np.clip(ne_mean - ne_std, 0, 500), np.clip(ne_mean + ne_std, 0, 500), alpha=0.2, color="tab:orange")
    ax2.axhline(475, color="tab:green", linestyle="--", linewidth=1.3, label="Solved Threshold (475)")
    ax2.set_title("Deterministic Evaluation Return (Every 20 Episodes)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Eval Return (mean ± std)")
    ax2.set_ylim(0, 530)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_no_target.png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 5 DQN CartPole-v1 experiments.")
    parser.add_argument("--mode", choices=["baseline", "no_target", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    baseline = None
    no_target = None

    if args.mode in ("baseline", "all"):
        baseline = run_condition("baseline", target_sync_freq=500, num_episodes=args.episodes, out_dir=args.out_dir)
        plot_baseline(baseline, args.out_dir)

    if args.mode in ("no_target", "all"):
        no_target = run_condition("no_target", target_sync_freq=1, num_episodes=args.episodes, out_dir=args.out_dir)

    if baseline is not None and no_target is not None:
        plot_comparison(baseline, no_target, args.out_dir)


if __name__ == "__main__":
    main()