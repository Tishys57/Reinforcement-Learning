import gymnasium as gym
import numpy as np


def epsilon_greedy_policy(Q, state, epsilon, rng):
  """Selects an action using an epsilon-greedy policy.

  Ties in argmax are broken uniformly at random.
  """
  if rng.random() < epsilon:
    return rng.integers(Q.shape[1])
  q_vals = Q[state]
  return rng.choice(np.flatnonzero(q_vals == q_vals.max()))


def q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma):
  """Performs an off-policy Q-learning update.

  `terminated` must be True only for true terminal states (goal or hole),
  NOT for time-limit truncation, so we still bootstrap on truncation.
  """
  target = reward if terminated else reward + gamma * Q[next_state].max()
  Q[state, action] += alpha * (target - Q[state, action])
  return Q


def sarsa_update(
    Q, state, action, reward, next_state, next_action, terminated, alpha, gamma
):
  """Performs an on-policy SARSA update."""
  target = reward if terminated else reward + gamma * Q[next_state, next_action]
  Q[state, action] += alpha * (target - Q[state, action])
  return Q


def train(
    env,
    n_episodes=100000,
    alpha=0.5,
    alpha_end=0.03,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_ratio=0.5,
    algorithm="q_learning",
    seed=0,
    snapshot_interval=1000,
):
  """Trains an agent using either Q-learning or SARSA on FrozenLake.

  alpha is the STARTING learning rate; it decays linearly to alpha_end
  over the full run.
  """
  rng = np.random.default_rng(seed)
  n_states, n_actions = env.observation_space.n, env.action_space.n
  Q = np.zeros((n_states, n_actions))

  episodic_returns = []
  episode_lengths = []
  q_snapshots = {}
  decay_steps = int(n_episodes * eps_decay_ratio)

  for episode in range(n_episodes):
    # Linear epsilon decay, then constant at eps_end
    epsilon = max(
        eps_end, eps_start - (episode / decay_steps) * (eps_start - eps_end)
    )
    # Linear alpha decay over the whole run
    alpha_t = max(alpha_end, alpha * (1 - episode / n_episodes))

    # Seed the environment only once, on the very first reset
    state, _ = env.reset(seed=seed if episode == 0 else None)
    done = False
    total_reward = 0.0
    steps = 0

    action = epsilon_greedy_policy(Q, state, epsilon, rng)

    while not done:
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      total_reward += reward
      steps += 1

      if algorithm == "q_learning":
        Q = q_learning_update(
            Q, state, action, reward, next_state, terminated, alpha_t, gamma
        )
        if not done:
          action = epsilon_greedy_policy(Q, next_state, epsilon, rng)
      elif algorithm == "sarsa":
        # Pick next action whenever the state is non-terminal (incl. truncation)
        next_action = (
            epsilon_greedy_policy(Q, next_state, epsilon, rng)
            if not terminated
            else None
        )
        Q = sarsa_update(
            Q, state, action, reward, next_state, next_action,
            terminated, alpha_t, gamma
        )
        action = next_action
      else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

      state = next_state

    episodic_returns.append(total_reward)
    episode_lengths.append(steps)

    if (episode + 1) % snapshot_interval == 0 or episode == 0:
      q_snapshots[episode + 1] = Q.copy()

  return Q, episodic_returns, episode_lengths, q_snapshots


def evaluate(env, Q, n_episodes=1000, seed=123):
  """Runs the purely greedy policy (epsilon = 0) and returns the success rate."""
  rng = np.random.default_rng(seed)
  successes = 0.0
  for episode in range(n_episodes):
    state, _ = env.reset(seed=seed if episode == 0 else None)
    done = False
    while not done:
      q_vals = Q[state]
      action = rng.choice(np.flatnonzero(q_vals == q_vals.max()))
      state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      successes += reward
  return successes / n_episodes