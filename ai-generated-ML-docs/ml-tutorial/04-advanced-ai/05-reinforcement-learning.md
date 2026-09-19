# 05 · Reinforcement Learning Essentials

> **Goal:** Understand RL well enough for interviews, RLHF, and simple agents. (Deep specialization is optional.)

## Core Setup
An **agent** in **state** `s` takes **action** `a`, gets **reward** `r`, moves to `s'`. Goal: maximize total (discounted) reward.

| Term | Meaning |
|---|---|
| Policy π(a\|s) | Strategy: what to do in each state |
| Value V(s) / Q(s,a) | Expected future reward |
| Discount γ | Importance of future rewards (0.9–0.99) |
| Exploration vs exploitation | Try new actions vs use the best known (ε-greedy) |
| MDP | Formal framework (states, actions, transitions, rewards) |

## Algorithm Families
| Family | Examples | Idea |
|---|---|---|
| Value-based | Q-learning, DQN | Learn Q, act greedily |
| Policy gradient | REINFORCE, **PPO** | Directly optimize the policy |
| Actor–critic | A2C, SAC | Policy + value together |
| Preference-based | RLHF (PPO), DPO, GRPO | Align LLMs with human/AI preferences |

## Q-Learning in 20 lines
```python
# pip install gymnasium
import numpy as np, gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, eps = 0.8, 0.95, 1.0

for ep in range(2000):
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample() if np.random.rand() < eps else np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])   # Bellman update
        s, done = s2, terminated or truncated
    eps = max(0.05, eps * 0.995)

print("Learned policy:", np.argmax(Q, axis=1).reshape(4, 4))
```

## Where RL Shows Up for AI Engineers
- **RLHF / DPO**: how chat LLMs are aligned.
- **Reasoning models**: RL on verifiable rewards (math, code tests).
- Robotics, games, recommendation, ad bidding.

## Exercises
1. Solve `CartPole-v1` with PPO using `stable-baselines3`.
2. Explain in your own words how DPO differs from RLHF with PPO.

---
Next → [Model Serving](../05-mlops/01-model-serving.md)
