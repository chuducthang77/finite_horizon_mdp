import numpy as np
from utils.softmax import softmax

# 3. Function to Calculate Optimal Value V*(s=0, h=0)
def calculate_optimal_value_function(env):
    states = range(env.num_states)
    H = env.horizon
    V = {h: {s: 0.0 for s in states} for h in range(H + 1)}

    terminal_state = env.terminal_state
    chain_end = env.chain_length - 1

    # Backward induction
    for h in reversed(range(H)):
        for s in range(env.chain_length):  # Skip the terminal state
            # Action 0: Terminate, reward = 0.5, no future value
            terminate_value = env.exit_reward

            # Action 1: Continue
            if s == chain_end:
                # Reaching terminal from last state yields reward = 1.0
                continue_reward = env.final_reward
                next_state = terminal_state
                continue_value = continue_reward  # No future value after terminal
            else:
                next_state = s + 1
                continue_reward = -env.intermediate_reward
                continue_value = continue_reward + V[h + 1][next_state]

            # Take the max between terminating and continuing
            V[h][s] = max(terminate_value, continue_value)

        # Terminal state stays at 0 value
        V[h][terminal_state] = 0.0

    V_mu  = sum(V[0][s] for s in range(env.chain_length)) / (env.chain_length)
    return V, V_mu