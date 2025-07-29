import numpy as np

# Chain MDP environment
class DeepSeaEnv:
    def __init__(self, horizon=10, grid_size = 5):
        if not isinstance(grid_size, int) or grid_size <= 1:
            raise ValueError("grid_size must be an integer greater than 1.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        self.grid_size = grid_size
        self.num_actions = 2
        self.num_states = self.grid_size * (self.grid_size + 1) // 2
        self.horizon = horizon

        self.start_state = (0, 0)
        self.treasure_state = (self.grid_size - 1, self.grid_size - 1)

        self.state = None
        self.done = False
        self.current_step = 0

        self.reset()

    def _state_to_idx(self, state):
        """Maps a (row, col) state to a unique integer index."""
        r, c = state
        if c > r:
            raise ValueError(f"Invalid state: column {c} cannot be greater than row {r}.")
        # The index is the count of states in rows above + the column index
        return (r * (r + 1) // 2) + c

    def reset(self):
        self.state = self.start_state
        self.current_step = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise Exception("Episode is finished. Please call reset() to start a new episode.")

        if action not in [0, 1]:
            raise ValueError("Invalid action. Choose 0 (left) or 1 (right).")

        self.current_step += 1
        current_row, current_col = self.state
        next_row = current_row + 1

        if action == 0:
            # Terminate immediately
            next_col = current_col
            reward = 0.0
        else:  # action == 1
            next_col = current_col + 1
            reward = -0.01 / self.grid_size

        self.state = (next_row, next_col)

        if self.state == self.treasure_state:
            reward = 1.0
            self.done = True
        elif next_row >= self.grid_size or next_col >= self.grid_size or self.current_step >= self.horizon:
            self.done = True

        return self.state, reward, self.done