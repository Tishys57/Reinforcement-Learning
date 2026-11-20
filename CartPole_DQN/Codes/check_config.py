import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dqn

torch.set_num_threads(1)
cpu = torch.device("cpu")


class TwoLayerQNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(F.relu(self.fc1(x)))


class XavierThreeLayerQNet(dqn.QNetwork):
    def _init_weights(self):
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


OrigQNet = dqn.QNetwork

tests = [
    ("2-layer (4->128->2), lr=1e-3, buf=50k, clip=None", TwoLayerQNet,
     {"lr": 1e-3, "hidden_dim": 128, "buffer_capacity": 50_000, "grad_clip": None}),
    ("2-layer (4->256->2), lr=1e-3, buf=50k, clip=None", TwoLayerQNet,
     {"lr": 1e-3, "hidden_dim": 256, "buffer_capacity": 50_000, "grad_clip": None}),
    ("3-layer (4->128->128->2), lr=1e-3, buf=50k, clip=None", OrigQNet,
     {"lr": 1e-3, "hidden_dim": 128, "buffer_capacity": 50_000, "grad_clip": None}),
    ("3-layer Xavier, lr=1e-3, buf=50k, clip=10.0", XavierThreeLayerQNet,
     {"lr": 1e-3, "hidden_dim": 128, "buffer_capacity": 50_000, "grad_clip": 10.0}),
]

for i, (label, net_cls, cfg) in enumerate(tests, 1):
    t0 = time.time()
    dqn.QNetwork = net_cls
    print(f"[{i}/{len(tests)}] {label}...", flush=True)
    logs = dqn.train_dqn(seed=0, num_episodes=500, device=cpu, **cfg)
    ret = np.array(logs["episode_returns"], dtype=np.float64)
    ma100 = np.convolve(ret, np.ones(100) / 100.0, mode="valid")
    hits = np.where(ma100 >= 475.0)[0]
    hit_ep = int(hits[0] + 100) if len(hits) > 0 else None
    print(
        f"  -> {time.time()-t0:5.1f}s | Best MA100: {np.max(ma100):6.2f} | "
        f"Hit 475 @ Ep: {hit_ep} | Last 50 avg: {np.mean(ret[-50:]):6.2f} | "
        f"Last 3 evals: {[round(x, 1) for x in logs['eval_returns'][-3:]]}",
        flush=True,
    )