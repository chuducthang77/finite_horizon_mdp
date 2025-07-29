from utils.softmax import softmax

def calculate_value_function(env, policy_params=None):
    grid_size = env.grid_size
    H = env.horizon
    V = {h: {} for h in range(H + 1)}

    for r in range(grid_size):
        for c in range(r + 1):
            V[H][(r, c)] = 0.0

    # Backward induction
    for h in reversed(range(H)):

        for c in reversed(range(h + 1)):  # skip terminal
            # Determine action probabilities
            state = (h, c)

            if state == env.treasure_state:
                V[h][state] = 1.0
                continue

            # For non-terminal states, calculate the Bellman expectation
            # Convert state (r, c) to a single integer index
            state_idx = env._state_to_idx(state)
            probabilities = softmax(policy_params[h][state_idx])

            # Value of Action 0 (Down)
            reward_down = 0.0
            next_state_down = (h + 1, c)
            future_value_down = V[h + 1].get(next_state_down, 0.0)
            value_down = reward_down + future_value_down

            # Value of Action 1 (Down-Right)
            reward_right = -0.01 / grid_size
            next_state_right = (h + 1, c + 1)
            future_value_right = V[h + 1].get(next_state_right, 0.0)
            value_right = reward_right + future_value_right

            # The state's value at time h is the expectation over actions
            V[h][state] = (probabilities[0] * value_down) + (probabilities[1] * value_right)

    # The overall value of the policy is the value of the start state at time 0
    V_start = V[0].get((0, 0), 0.0)

    return V, V_start