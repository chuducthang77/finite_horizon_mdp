import numpy as np

class DiceStopEnv:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.dice_faces = [1, 2, 3, 4, 5, 6]
        self.terminal_state = 6  # We’ll use index 6 for terminal state (Δ)
        self.num_states = 7  # 6 dice outcomes + terminal
        self.num_actions = 2  # 0 = continue, 1 = stop
        self.reset()

    def reset(self):
        self.current_state = np.random.choice(self.dice_faces) - 1  # 0-based index
        self.current_step = 0
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done:
            raise Exception("Episode has already ended.")

        reward = 0

        if action == 1:
            # Stop and move to terminal
            reward = self.current_state + 1  # state index + 1 equals dice value
            self.current_state = self.terminal_state
            self.done = True
        elif action == 0:
            # Continue: resample a dice value
            self.current_state = np.random.choice(self.dice_faces) - 1
        else:
            raise ValueError("Invalid action. Must be 0 (continue) or 1 (stop)")

        self.current_step += 1
        if self.current_step >= self.horizon or self.done:
            self.done = True

        return self.current_state, reward, self.done

    def sample_action(self):
        return np.random.choice([0, 1])

    def is_terminal(self, state):
        return state == self.terminal_state
