import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Chain MDP environment
class ChainEnv:
    def __init__(self, chain_length=5, horizon=10):
        self.chain_length = chain_length
        self.horizon = horizon
        self.num_states = chain_length + 1  # Last state is terminal
        self.num_actions = 2  # 0: terminate, 1: continue
        self.terminal_state = self.chain_length
        self.reset()

    def reset(self):
        self.current_state = 0
        self.current_step = 0
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done or self.current_step >= self.horizon:
            raise Exception("Episode already finished.")

        if action not in [0, 1]:
            raise ValueError("Invalid action")

        if action == 0:
            # Terminate immediately
            next_state = self.terminal_state
            reward = 0.5
        else:  # action == 1
            next_state = self.current_state + 1
            if next_state >= self.chain_length:
                next_state = self.terminal_state

            # Reward 1 only if we reach the terminal state from the last state
            reward = 1.0 if (self.current_state == self.chain_length - 1 and action == 1) else 0.0

        self.current_state = next_state
        self.current_step += 1
        self.done = self.current_state == self.terminal_state

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
def calculate_optimal_value_start(env):
    states = range(env.num_states)
    H = env.horizon
    V = {h: {s: 0.0 for s in states} for h in range(H + 1)}

    terminal_state = env.terminal_state
    chain_end = env.chain_length - 1

    # Backward induction
    for h in reversed(range(H)):
        for s in range(env.chain_length):  # Skip the terminal state
            # Action 0: Terminate, reward = 0.5, no future value
            terminate_value = 0.5

            # Action 1: Continue
            if s == chain_end:
                # Reaching terminal from last state yields reward = 1.0
                continue_reward = 1.0
                next_state = terminal_state
                continue_value = continue_reward  # No future value after terminal
            else:
                next_state = s + 1
                continue_reward = 0.0
                continue_value = continue_reward + V[h + 1][next_state]

            # Take the max between terminating and continuing
            V[h][s] = max(terminate_value, continue_value)

        # Terminal state stays at 0 value
        V[h][terminal_state] = 0.0

    return V


# Example usage:
env = ChainEnv(chain_length=5, horizon=10)
V_star = calculate_optimal_value_start(env)
print(V_star[0][0])  # V*(s=0, h=0)


def calculate_value_start(env, policy_params=None):
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
            value_terminate = 0.5

            # Action 1: continue
            if s == chain_end:
                # last state before terminal: reward 1.0 and ends
                value_continue = 1.0
            else:
                next_state = s + 1
                value_continue = 0.0 + V[h + 1][next_state]

            # Bellman expectation over actions
            V[h][s] = probabilities[0] * value_terminate + probabilities[1] * value_continue

        # Terminal state's value is always 0
        V[h][terminal_state] = 0.0

    return V


# Hyperparameters
chain_len = 4
horizon = chain_len + 2  # Needs slightly more steps than chain len to reach end
# learning_rates = [0.1, 0.5, 1., 2.]
learning_rates = [2.]
num_episodes = 100000
print_every = 200
n_runs = 30

# Initialization
env = ChainEnv(chain_length=chain_len, horizon=horizon)
num_states = env.num_states
num_actions = env.num_actions

optimal_value = calculate_optimal_value_start(env)[0][0]
print(f"Optimal Value V*(s=0, h=0) = {optimal_value}")
print("-" * 30)

results_by_lr = {}

for lr in learning_rates:
    print(f"\n=== Averaging {n_runs} runs for LR={lr} ===")
    all_runs_suboptimalities = np.zeros((n_runs, num_episodes))

    for run in range(n_runs):
        # --- Policy Parameters ---
        # A list where each element is the parameter table (logits) for that time step
        policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
        suboptimality_history = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            total_reward = 0

            for h in range(horizon):
                action, probs = get_action(state, h, policy_params, num_actions)
                next_state, reward, done = env.step(action)  # Updated: get 'done' flag
                episode_data.append({"state": state, "action": action, "reward": reward, "probs": probs})
                state = next_state
                total_reward += reward
                if done:  # Terminate early if the environment ends
                    break

            # Calculate returns-to-go (G_h) for the episode
            returns_to_go = np.zeros(len(episode_data))
            cumulative_return = 0
            for h in reversed(range(len(episode_data))):
                cumulative_return = episode_data[h]["reward"] + cumulative_return
                returns_to_go[h] = cumulative_return

            # --- Policy Update ---
            for h in range(len(episode_data)):
                step_data = episode_data[h]
                s_h = step_data["state"]
                a_h = step_data["action"]
                G_h = returns_to_go[h]
                probs_h = step_data["probs"]

                grad_log_pi = np.zeros(num_actions)
                grad_log_pi[a_h] = 1.0
                grad_log_pi -= probs_h

                policy_params[h][s_h, :] += lr * G_h * grad_log_pi

            result = calculate_value_start(env, policy_params)[0][0]
            suboptimality = optimal_value - result
            suboptimality_history.append(suboptimality)

        all_runs_suboptimalities[run] = suboptimality_history

    averaged_suboptimality = np.mean(all_runs_suboptimalities, axis=0)
    np.save('averaged_suboptimality.npy', averaged_suboptimality)
    np.save('all_runs_suboptimalities.npy', all_runs_suboptimalities)

    std_suboptimality = np.std(all_runs_suboptimalities, axis=0)
    episodes = np.arange(1, num_episodes + 1)
    # Save results
    results_by_lr[lr] = {
        "avg_suboptimality": averaged_suboptimality,
        "std_suboptimality": std_suboptimality,
        "episodes": episodes,
        "all_runs_suboptimalities": all_runs_suboptimalities
    }

    print(f"Finished training with learning rate {lr}.")

episodes = np.arange(1, num_episodes + 1)
for n in range(n_runs):
    plt.figure(figsize=(10, 6))
    for lr, data in results_by_lr.items():
        plt.plot(episodes, data["all_runs_suboptimalities"][n, :], label=f"LR={lr}", linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel(r"Suboptimality: $V^*_{0}(s_0) - V_{\theta_T}(s_0)$")
    plt.title(f"Suboptimality with run {n} with {lr} learning rate")
    plt.grid(True)
    plt.yscale('log')
    plt.legend(title="Learning Rate")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"suboptimality_chain_mdp_run_{n}_{timestamp}.png")
    # plt.show()

print("-" * 30)
print("Training finished.")


for lr, data in results_by_lr.items():
    episodes = data["episodes"]
    mean_vals = data["avg_suboptimality"]
    std_vals = data["std_suboptimality"]

    plt.plot(episodes, mean_vals, label=f"LR={lr}", linewidth=1.5)
    plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)

plt.xlabel("Episode")
plt.ylabel(r"Average Suboptimality: $V^*_{0}(s_0) - V_{\theta_T}(s_0)$")
plt.title(f"Average Suboptimality with Std Dev ({n_runs} Runs per LR)")
plt.grid(True)
plt.yscale('log')
plt.legend(title="Learning Rate")
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"suboptimality_chain_mdp_{timestamp}.png")
# plt.show()
