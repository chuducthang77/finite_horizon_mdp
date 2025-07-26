from utils.softmax import softmax

def calculate_value_function(env, policy_params=None):
    states = range(env.num_states)
    H = env.horizon
    V = {h: {s: 0.0 for s in states} for h in range(H + 1)}

    terminal_state = env.terminal_state
    chain_end = env.chain_length - 1

    # Backward induction
    for h in reversed(range(H)):
        for s in range(env.chain_length):  # skip terminal
            # Determine action probabilities
            if policy_params is not None:
                probabilities = softmax(policy_params[h][s])

            # Action 0: terminate → reward 0.5, no future value
            value_terminate = env.exit_reward

            # Action 1: continue
            if s == chain_end:
                # last state before terminal: reward 1.0 and ends
                value_continue = env.final_reward
            else:
                next_state = s + 1
                value_continue = env.intermediate_reward + V[h + 1][next_state]

            # Bellman expectation over actions
            V[h][s] = probabilities[0] * value_terminate + probabilities[1] * value_continue

        # Terminal state's value is always 0
        V[h][terminal_state] = 0.0

    V_mu = sum(V[0][s] for s in range(env.chain_length)) / (env.chain_length)
    return V, V_mu