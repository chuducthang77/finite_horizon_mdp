import numpy as np
from utils.softmax import softmax

def calculate_optimal_value_function(env):
    states = range(env.num_states)  # includes terminal state Δ
    H = env.horizon
    V = {h: {s: 0.0 for s in states} for h in range(H + 1)}

    terminal_state = env.terminal_state
    num_faces = len(env.dice_faces)

    for h in reversed(range(H)):
        for s in range(env.num_states):
            if s == terminal_state:
                V[h][s] = 0.0
                continue

            # Action 1: Stop (receive reward = dice face, then go to terminal)
            stop_reward = s + 1  # since dice face = state index + 1
            stop_value = stop_reward  # terminal gives 0 future reward

            # Action 0: Continue (uniform over 6 dice values)
            expected_future_value = 0
            for next_state in range(num_faces):  # 0 to 5
                expected_future_value += V[h + 1][next_state]
            expected_future_value /= num_faces
            continue_value = expected_future_value

            # Optimal value is the better of stop or continue
            V[h][s] = max(stop_value, continue_value)

    V_mu  = sum(V[0][s] for s in range(env.num_states)) / (env.num_states)

    return V, V_mu
