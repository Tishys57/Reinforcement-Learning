import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Task 1: Fixed-Capacity Circular Replay Buffer (15 pts)
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int = 4) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.ptr] = 1.0 if bool(done) else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones


# =====================================================================
# Task 2: Q-Network (10 pts)
# =====================================================================
class QNetwork(nn.Module):
    def __init__(
        self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# =====================================================================
# Task 3: DQN Loss (15 pts)
# =====================================================================
def compute_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    gamma: float = 0.99,
) -> torch.Tensor:
    states, actions, rewards, next_states, dones = batch
    device = next(online_net.parameters()).device

    states = states.to(device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.int64)
    rewards = rewards.to(device=device, dtype=torch.float32)
    next_states = next_states.to(device=device, dtype=torch.float32)
    dones = dones.to(device=device, dtype=torch.float32)

    target_net.eval()
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1).values
        targets = rewards + gamma * (1.0 - dones) * max_next_q

    q_values = online_net(states)
    q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    loss = F.mse_loss(q_sa, targets)
    return loss


# =====================================================================
# Evaluation Helper (Deterministic Policy)
# =====================================================================
def evaluate_policy(
    online_net: nn.Module,
    env_name: str = "CartPole-v1",
    num_episodes: int = 5,
    seed: int = 1000,
    device: Optional[torch.device] = None,
) -> float:
    if device is None:
        device = next(online_net.parameters()).device

    was_training = online_net.training
    online_net.eval()
    eval_env = gym.make(env_name)
    returns: List[float] = []

    with torch.no_grad():
        for i in range(num_episodes):
            state, _ = eval_env.reset(seed=seed + i)
            done = False
            ep_return = 0.0
            while not done:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(online_net(state_t).argmax(dim=1).item())
                state, reward, terminated, truncated, _ = eval_env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
            returns.append(ep_return)

    eval_env.close()
    if was_training:
        online_net.train()
    return float(np.mean(returns))


# =====================================================================
# Task 4: Training Loop (20 pts)
# =====================================================================
def train_dqn(
    env_name: str = "CartPole-v1",
    seed: int = 0,
    num_episodes: int = 500,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    warmup_steps: int = 1_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 10_000,
    target_sync_freq: int = 500,
    eval_freq: int = 20,
    eval_episodes: int = 5,
    grad_clip: Optional[float] = 10.0,
    device: Optional[torch.device] = None,
    verify_sync: bool = False,
) -> Dict[str, List[float]]:
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)

    episode_returns: List[float] = []
    losses: List[float] = []
    mean_max_qs: List[float] = []
    epsilons: List[float] = []
    eval_returns: List[float] = []
    eval_episodes_idx: List[int] = []

    global_step = 0

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed if ep == 1 else None)
        ep_return = 0.0
        done = False

        while not done:
            if global_step < warmup_steps:
                eps = eps_start
            else:
                decay_progress = min(1.0, (global_step - warmup_steps) / float(eps_decay_steps))
                eps = eps_start - decay_progress * (eps_start - eps_end)

            epsilons.append(eps)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online_net(state_t).argmax(dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, float(reward), next_state, terminated)

            state = next_state
            ep_return += float(reward)
            global_step += 1

            if global_step > warmup_steps and len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size, device=device)
                loss = compute_dqn_loss(online_net, target_net, batch, gamma=gamma)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(online_net.parameters(), grad_clip)
                optimizer.step()

                with torch.no_grad():
                    batch_max_q = online_net(batch[0]).max(dim=1).values.mean().item()

                if not np.isfinite(batch_max_q):
                    raise FloatingPointError(f"Non-finite Q-value detected at step {global_step}.")

                losses.append(float(loss.item()))
                mean_max_qs.append(float(batch_max_q))

            if global_step % target_sync_freq == 0:
                target_net.load_state_dict(online_net.state_dict())
                if verify_sync:
                    for p_on, p_tgt in zip(online_net.parameters(), target_net.parameters()):
                        assert torch.equal(p_on, p_tgt), "Target sync mismatch!"

        episode_returns.append(ep_return)

        if ep % eval_freq == 0:
            eval_ret = evaluate_policy(
                online_net,
                env_name=env_name,
                num_episodes=eval_episodes,
                seed=10_000 + seed * 100 + ep,
                device=device,
            )
            eval_returns.append(eval_ret)
            eval_episodes_idx.append(ep)

    env.close()
    return {
        "episode_returns": episode_returns,
        "losses": losses,
        "mean_max_q": mean_max_qs,
        "epsilons": epsilons,
        "eval_returns": eval_returns,
        "eval_episodes": eval_episodes_idx,
    }