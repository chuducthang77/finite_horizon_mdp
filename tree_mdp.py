import numpy as np
import matplotlib.pyplot as plt

class TreeEnv:
    def __init__(self):
        self.reset()

        # Define the deterministic transition structure as a dict:
        # (current_state, action) -> (next_state, reward)
        self.transitions = {
            # Root state 0
            (0, 0): (1, 1),    # Top branch
            (0, 1): (2, 2),    # Middle branch
            (0, 2): (3, 0.5),  # Bottom branch

            # State 1 children
            (1, 0): (4, 2),
            (1, 1): (5, 0),
            (1, 2): (6, 0),

            # State 2 children
            (2, 0): (7, 1),
            (2, 1): (8, 0),
            (2, 2): (9, 0),

            # State 3 children
            (3, 0): (10, 0.5),
            (3, 1): (11, 0),
            (3, 2): (12, 0),
        }

        self.terminal_states = set(range(4, 13))  # Leaf nodes

    def reset(self):
        self.current_state = 0
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done:
            raise Exception("Episode already finished.")
        key = (self.current_state, action)
        if key not in self.transitions:
            raise ValueError(f"Invalid action {action} at state {self.current_state}")

        next_state, reward = self.transitions[key]
        self.current_state = next_state
        self.done = next_state in self.terminal_states
        return next_state, reward, self.done

# 2. Softmax Policy (Non-Stationary)
def softmax(logits):
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def get_action(state, step_h, policy_params, num_actions):
    """Selects an action using the policy for the specific step h."""
    if step_h >= len(policy_params):
        # Handle case if trying to get action beyond horizon (shouldn't happen in normal loop)
        print(f"Warning: Step {step_h} is outside policy horizon {len(policy_params)}. Using last step policy.")
        step_h = len(policy_params) - 1

    logits = policy_params[step_h][state]
    probabilities = softmax(logits)
    action = np.random.choice(num_actions, p=probabilities)
    return action, probabilities


# 3. Function to Calculate Optimal Value V*(s=0, h=0)
def calculate_optimal_value_start(env, horizon):
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

def calculate_value_tree(env, policy_params=None, horizon=2):
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


# --- Setup ---
env = TreeEnv()
states = list(set(s for s, _ in env.transitions.keys()) | set(s for s, _ in env.transitions.values()))
state_to_idx = {s: i for i, s in enumerate(states)}  # consistent indexing
idx_to_state = {i: s for s, i in state_to_idx.items()}

num_states = len(states)
num_actions = 3
horizon = 2
learning_rates = [0.1, 0.5, 1., 2., 1e-3, 1e-5]
num_episodes = 1000
n_runs = 1

# Optimal value
optimal_value, optimal_v0 = calculate_optimal_value_start(env, horizon)
print(f"Optimal Value V*(s=0, h=0) = {optimal_value[0][0]}")

# Results storage
results_by_lr = {
    lr: {
        "all_runs_suboptimalities": np.zeros((n_runs, num_episodes)),
        "all_v_histories": np.zeros((n_runs, num_episodes, horizon, num_states)),
        "all_prob_action_histories": np.zeros((n_runs, num_episodes, horizon, num_states, num_actions)),
    } for lr in learning_rates
}

for run in range(n_runs):
    print(f"\n==== Run {run+1} ===")
    for lr in learning_rates:
        print(f"Training with LR={lr}")

        policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]

        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            total_reward = 0

            for h in range(horizon):
                state_idx = state_to_idx[state]
                logits = policy_params[h][state_idx]
                probs = softmax(logits)
                action = np.random.choice(num_actions, p=probs)

                next_state, reward, done = env.step(action)
                episode_data.append({
                    "state": state_idx,
                    "action": action,
                    "reward": reward,
                    "probs": probs
                })

                state = next_state
                total_reward += reward
                if done:
                    break

            # Compute returns
            returns_to_go = np.zeros(len(episode_data))
            G = 0
            for h in reversed(range(len(episode_data))):
                G = episode_data[h]["reward"] + G
                returns_to_go[h] = G

            # Policy gradient update
            for h in range(len(episode_data)):
                s_idx = episode_data[h]["state"]
                a = episode_data[h]["action"]
                G_h = returns_to_go[h]
                probs = episode_data[h]["probs"]

                grad_log_pi = -probs
                grad_log_pi[a] += 1
                policy_params[h][s_idx] += lr * G_h * grad_log_pi

            # Evaluate value function
            value_function, V0 = calculate_value_tree(env, policy_params, horizon=horizon)
            subopt = optimal_v0 - V0

            results_by_lr[lr]["all_runs_suboptimalities"][run, episode] = subopt

            for h in range(horizon):
                for s_idx in range(num_states):
                    s = idx_to_state[s_idx]
                    results_by_lr[lr]["all_v_histories"][run, episode, h, s_idx] = value_function[h][s]
                    probs = softmax(policy_params[h][s_idx])
                    results_by_lr[lr]["all_prob_action_histories"][run, episode, h, s_idx] = probs

# Post-processing (mean/std)
for lr in learning_rates:
    subopt = results_by_lr[lr]["all_runs_suboptimalities"]
    results_by_lr[lr]["avg_suboptimality"] = np.mean(subopt, axis=0)
    results_by_lr[lr]["std_suboptimality"] = np.std(subopt, axis=0)

    results_by_lr[lr]["mean_v"] = np.mean(results_by_lr[lr]["all_v_histories"], axis=0)
    results_by_lr[lr]["std_v"] = np.std(results_by_lr[lr]["all_v_histories"], axis=0)

    results_by_lr[lr]["mean_prob_action"] = np.mean(results_by_lr[lr]["all_prob_action_histories"], axis=0)
    results_by_lr[lr]["std_prob_action"] = np.std(results_by_lr[lr]["all_prob_action_histories"], axis=0)

    results_by_lr[lr]["episodes"] = np.arange(1, num_episodes + 1)
