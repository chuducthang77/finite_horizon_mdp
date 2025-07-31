import gym
import numpy as np

class CartPoleDiscreteEnv:
    def __init__(self, bins=(6, 6, 6, 6), horizon=200):
        self.env = gym.make('CartPole-v1')
        self.horizon = horizon
        self.bins = bins
        self.num_actions = self.env.action_space.n
        self.state_bins = [
            np.linspace(-4.8, 4.8, bins[0] - 1),
            np.linspace(-4, 4, bins[1] - 1),
            np.linspace(-0.418, 0.418, bins[2] - 1),
            np.linspace(-4, 4, bins[3] - 1),
        ]
        self.num_states = np.prod([b + 1 for b in bins])

    def reset(self):
        obs, _ = self.env.reset()
        return tuple(np.digitize(obs[i], self.state_bins[i]) for i in range(4))

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = tuple(np.digitize(obs[i], self.state_bins[i]) for i in range(4))
        return state, reward, done

    def _state_to_idx(self, state):
        idx = 0
        for i, s in enumerate(state):
            idx *= self.bins[i]
            idx += s
        return idx
