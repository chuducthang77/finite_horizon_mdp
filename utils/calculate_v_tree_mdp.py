from utils.softmax import softmax
import numpy as np

def calculate_value_function(env, policy_params=None, horizon=2):
    # Gather all reachable states
    states = list(set(s for s, _ in env.transitions.keys()) | set(s for s, _ in env.transitions.values()))
    V = {h: {s: 0.0 for s in states} for h in range(horizon + 1)}

    # Backward induction
    for h in reversed(range(horizon)):
        for s in states:
            if s in env.terminal_states:
                V[h][s] = 0.0
                continue

            # Default: uniform policy over available actions
            available_actions = [a for a in [0, 1, 2] if (s, a) in env.transitions]
            if policy_params is not None:
                logits = policy_params[h][s]
                probs = softmax(logits)
                probs = [probs[a] if a in available_actions else 0.0 for a in range(3)]
                probs = np.array(probs)
                probs /= probs.sum()  # re-normalize in case some actions are invalid
            else:
                probs = np.ones(len(available_actions)) / len(available_actions)
                probs = [probs[available_actions.index(a)] if a in available_actions else 0.0 for a in range(3)]

            # Bellman expectation
            expected_value = 0.0
            for a in range(3):
                if (s, a) in env.transitions:
                    next_state, reward = env.transitions[(s, a)]
                    expected_value += probs[a] * (reward + V[h + 1][next_state])

            V[h][s] = expected_value

    V_start = V[0][0]  # Assuming starting state is 0
    return V, V_start