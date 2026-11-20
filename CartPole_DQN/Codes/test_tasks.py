import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

from dqn import QNetwork, ReplayBuffer, compute_dqn_loss, train_dqn

torch.set_num_threads(1)


def test_task1_replay_buffer() -> None:
    print("[Test 1] Checking ReplayBuffer capacity, done conversion, and 10^6 uniformity...")
    buf = ReplayBuffer(capacity=5, state_dim=4)
    for i in range(8):
        s = np.full(4, float(i), dtype=np.float32)
        ns = np.full(4, float(i + 1), dtype=np.float32)
        buf.push(s, i % 2, float(i), ns, done=(i % 2 == 1))

    assert len(buf) == 5, f"Expected length 5, got {len(buf)}"
    stored_ids = set(buf.rewards[: len(buf)].tolist())
    assert stored_ids == {3.0, 4.0, 5.0, 6.0, 7.0}, f"Unexpected contents: {stored_ids}"

    _, _, _, _, dones = buf.sample(32)
    assert dones.dtype == torch.float32
    assert set(dones.tolist()).issubset({0.0, 1.0})

    cap = 10
    ubuf = ReplayBuffer(capacity=cap, state_dim=4)
    for i in range(cap):
        ubuf.push(np.full(4, float(i)), 0, float(i), np.zeros(4), False)

    counts = np.zeros(cap, dtype=np.int64)
    for _ in range(1000):
        _, _, rew, _, _ = ubuf.sample(1000)
        vals, cnts = np.unique(rew.numpy().astype(np.int64), return_counts=True)
        counts[vals] += cnts

    assert counts.sum() == 1_000_000
    expected = 1_000_000 / cap
    max_rel_err = np.max(np.abs(counts - expected) / expected)
    assert max_rel_err < 0.02, f"Non-uniform sampling detected: {counts}"
    print(f"  PASS (max relative error over 10^6 draws = {max_rel_err:.4%})")


def test_task2_q_network() -> None:
    print("[Test 2] Checking QNetwork architecture and output...")
    net = QNetwork(state_dim=4, action_dim=2, hidden_dim=128)
    x = torch.randn(16, 4)
    out = net(x)
    assert out.shape == (16, 2)
    assert not torch.allclose(out.sum(dim=1), torch.ones(16)), "Output looks like softmax!"
    print("  PASS")


def test_task3_dqn_loss() -> None:
    print("[Test 3] Checking DQN loss precision (<1e-6) and zero target gradients...")
    torch.manual_seed(42)
    online = QNetwork(4, 2, 16)
    target = QNetwork(4, 2, 16)
    target.load_state_dict(online.state_dict())

    states = torch.randn(8, 4)
    actions = torch.randint(0, 2, (8,))
    rewards = torch.randn(8)
    next_states = torch.randn(8, 4)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    gamma = 0.99

    loss = compute_dqn_loss(online, target, (states, actions, rewards, next_states, dones), gamma=gamma)

    with torch.no_grad():
        q_next = target(next_states).max(dim=1).values
        y = rewards + gamma * (1.0 - dones) * q_next
    q_sa = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    expected_loss = F.mse_loss(q_sa, y)

    diff = abs(loss.item() - expected_loss.item())
    assert diff < 1e-6, f"Loss mismatch: {diff}"

    loss.backward()
    for p in online.parameters():
        assert p.grad is not None, "Online network received no gradients!"
    for p in target.parameters():
        assert p.grad is None, "Target network received gradients!"
    assert not target.training, "Target network must be in eval mode!"
    print(f"  PASS (loss diff = {diff:.2e}, target grads = None)")


def test_task4_training_seed0() -> None:
    print("[Test 4] Running train_dqn(seed=0, num_episodes=500)...")
    logs = train_dqn(seed=0, num_episodes=500, device=torch.device("cpu"), verify_sync=True)
    returns = np.array(logs["episode_returns"], dtype=np.float64)
    ma100 = np.convolve(returns, np.ones(100) / 100.0, mode="valid")
    best_ma100 = float(np.max(ma100))

    for ep in range(50, 501, 50):
        avg50 = float(np.mean(returns[ep - 50 : ep]))
        print(f"  Ep {ep:3d} | 50-ep avg return = {avg50:6.2f}")
    print(f"  Eval returns (every 20 eps): {[round(x, 1) for x in logs['eval_returns']]}")
    print(f"  Mean max-Q range: [{min(logs['mean_max_q']):.2f}, {max(logs['mean_max_q']):.2f}]")

    assert np.all(np.isfinite(logs["mean_max_q"])), "NaN/Inf in Q-values!"
    assert best_ma100 >= 400.0, f"Best 100-ep MA ({best_ma100:.2f}) < 400!"
    print(f"  PASS (Best 100-episode MA = {best_ma100:.2f})")


if __name__ == "__main__":
    test_task1_replay_buffer()
    test_task2_q_network()
    test_task3_dqn_loss()
    test_task4_training_seed0()
    print("\nAll Task 1-4 checks passed!")