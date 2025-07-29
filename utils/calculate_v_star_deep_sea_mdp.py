import numpy as np
from utils.softmax import softmax

# 3. Function to Calculate Optimal Value V*(s=0, h=0)
def calculate_optimal_value_function(env):
    grid_size = env.grid_size
    H = env.horizon

    # Value function is indexed by time step 'h' and state '(r, c)'
    V = {h: {} for h in range(H + 1)}

    # Base Case: At the horizon (h=H), the episode is over. Future value is 0.
    for r in range(grid_size):
        for c in range(r + 1):
            V[H][(r, c)] = 0.0

    # --- Backward Induction from H-1 down to 0 ---
    for h in reversed(range(H)):
        # At time step h, the agent must be in row h.
        # Iterate over all possible columns 'c' for row 'h'.
        for c in reversed(range(h + 1)):
            state = (h, c)

            # If the state is the treasure, its value is fixed at 1.0
            if state == env.treasure_state:
                V[h][state] = 1.0
                continue

            # --- For non-terminal states, calculate the optimal Q-values ---

            # Q-value for Action 0 (Down)
            reward_down = 0.0
            next_state_down = (h + 1, c)
            future_value_down = V[h + 1].get(next_state_down, 0.0)
            q_value_down = reward_down + future_value_down

            # Q-value for Action 1 (Down-Right)
            reward_right = -0.01 / grid_size
            next_state_right = (h + 1, c + 1)
            future_value_right = V[h + 1].get(next_state_right, 0.0)
            q_value_right = reward_right + future_value_right

            # The optimal value V* is the max over all action Q-values
            V[h][state] = max(q_value_down, q_value_right)

    # The optimal value of the environment is the value of the start state at time 0
    V_start = V[0].get((0, 0), 0.0)

    return V, V_start