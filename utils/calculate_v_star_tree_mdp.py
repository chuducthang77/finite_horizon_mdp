def calculate_optimal_value_function(env, horizon):
    states = list(set(s for s, _ in env.transitions.keys()) | set(s for s, _ in env.transitions.values()))
    V = {h: {s: 0.0 for s in states} for h in range(horizon + 1)}

    # Backward induction
    for h in reversed(range(horizon)):
        for s in states:
            if s in env.terminal_states:
                V[h][s] = 0.0
                continue

            action_values = []
            for a in [0, 1, 2]:
                key = (s, a)
                if key in env.transitions:
                    next_state, reward = env.transitions[key]
                    value = reward + V[h + 1][next_state]
                    action_values.append(value)

            if action_values:
                V[h][s] = max(action_values)

    V_start = V[0][0]  # Assuming start state is always 0
    return V, V_start
