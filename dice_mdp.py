import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DiceStopEnv:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.dice_faces = [1, 2, 3, 4, 5, 6]
        self.terminal_state = 6  # We’ll use index 6 for terminal state (Δ)
        self.num_states = 7  # 6 dice outcomes + terminal
        self.num_actions = 2  # 0 = continue, 1 = stop
        self.reset()

    def reset(self):
        self.current_state = np.random.choice(self.dice_faces) - 1  # 0-based index
        self.current_step = 0
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done:
            raise Exception("Episode has already ended.")

        reward = 0

        if action == 1:
            # Stop and move to terminal
            reward = self.current_state + 1  # state index + 1 equals dice value
            self.current_state = self.terminal_state
            self.done = True
        elif action == 0:
            # Continue: resample a dice value
            self.current_state = np.random.choice(self.dice_faces) - 1
        else:
            raise ValueError("Invalid action. Must be 0 (continue) or 1 (stop)")

        self.current_step += 1
        if self.current_step >= self.horizon or self.done:
            self.done = True

        return self.current_state, reward, self.done

    def sample_action(self):
        return np.random.choice([0, 1])

    def is_terminal(self, state):
        return state == self.terminal_state
    
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

def calculate_optimal_value_start(env):
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

    return V

def calculate_value_start(env, policy_params=None, pi=None):
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

    return V


learning_rates = [0.1, 0.5, 1.0, 2.0]
num_episodes = 100000
n_runs = 30

horizon = 5
env = DiceStopEnv(horizon=horizon)  # Replace with your actual class
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
        policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
        suboptimality_history = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            total_reward = 0

            for h in range(horizon):
                action, probs = get_action(state, h, policy_params, num_actions)
                next_state, reward, done = env.step(action)
                episode_data.append({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "probs": probs
                })
                state = next_state
                total_reward += reward
                if done:
                    break

            # Returns-to-go
            returns_to_go = np.zeros(len(episode_data))
            cumulative_return = 0
            for h in reversed(range(len(episode_data))):
                cumulative_return = episode_data[h]["reward"] + cumulative_return
                returns_to_go[h] = cumulative_return

            # Policy Gradient Update
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

            # Evaluation
            result = calculate_value_start(env, policy_params)[0][0]
            suboptimality = optimal_value - result
            suboptimality_history.append(suboptimality)

        all_runs_suboptimalities[run] = suboptimality_history

    # Average and save
    averaged_suboptimality = np.mean(all_runs_suboptimalities, axis=0)
    std_suboptimality = np.std(all_runs_suboptimalities, axis=0)
    episodes = np.arange(1, num_episodes + 1)

    results_by_lr[lr] = {
        "episodes": episodes,
        "avg_suboptimality": averaged_suboptimality,
        "std_suboptimality": std_suboptimality,
        "all_runs_suboptimalities": all_runs_suboptimalities
    }

    print(f"Done with LR={lr}")

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
    plt.savefig(f"suboptimality_dice_mdp_run_{n}_{timestamp}.png")


plt.figure(figsize=(10, 6))

for lr, result in results_by_lr.items():
    episodes = result["episodes"]
    avg = result["avg_suboptimality"]
    std = result["std_suboptimality"]

    plt.plot(episodes, avg, label=f"LR={lr}")
    plt.fill_between(episodes, avg - std, avg + std, alpha=0.2)

plt.xlabel("Episode")
plt.ylabel(r"Suboptimality: $V^*(s_0, h=0) - V_{\theta}(s_0, h=0)$")
plt.title("Suboptimality Gap vs Training Episodes")
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"suboptimality_dice_mdp_{timestamp}.png")
# plt.show()