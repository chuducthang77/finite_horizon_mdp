from utils.softmax import softmax

def calculate_value_function(env, policy_params=None, pi=None):
    states = range(env.num_states)
    H = env.horizon
    V = {h: {s: 0.0 for s in states} for h in range(H + 1)}

    terminal_state = env.terminal_state
    num_faces = len(env.dice_faces)  # 6

    for h in reversed(range(H)):
        for s in range(env.num_states):
            if s == terminal_state:
                V[h][s] = 0.0
                continue

            # Get policy probabilities
            if policy_params is not None:
                probabilities = softmax(policy_params[h][s])
            elif pi is not None:
                probabilities = pi[h][s]
            else:
                raise ValueError("Must provide either policy_params or pi.")

            # Action 0: continue — uniform over dice faces
            expected_value_continue = sum(V[h + 1][s_next] for s_next in range(num_faces)) / num_faces

            # Action 1: stop — reward = current face (s + 1), jump to terminal (no future reward)
            value_stop = s + 1

            # Expected value under current policy
            V[h][s] = probabilities[0] * expected_value_continue + probabilities[1] * value_stop

        V[h][terminal_state] = 0.0

    V_mu  = sum(V[0][s] for s in range(env.num_states)) / (env.num_states)
    return V, V_mu
