import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle


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
        self.current_state = np.random.randint(0, self.chain_length)
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
            reward = 7.0 if (self.current_state == self.chain_length - 1 and action == 1) else -0.5

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
                continue_reward = 7.0
                next_state = terminal_state
                continue_value = continue_reward  # No future value after terminal
            else:
                next_state = s + 1
                continue_reward = -0.5
                continue_value = continue_reward + V[h + 1][next_state]

            # Take the max between terminating and continuing
            V[h][s] = max(terminate_value, continue_value)

        # Terminal state stays at 0 value
        V[h][terminal_state] = 0.0

    V_mu  = sum(V[0][s] for s in range(env.chain_length)) / (env.chain_length)
    return V, V_mu


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
                value_continue = 7.0
            else:
                next_state = s + 1
                value_continue = -0.5 + V[h + 1][next_state]

            # Bellman expectation over actions
            V[h][s] = probabilities[0] * value_terminate + probabilities[1] * value_continue

        # Terminal state's value is always 0
        V[h][terminal_state] = 0.0
    
    V_mu  = sum(V[0][s] for s in range(env.chain_length)) / (env.chain_length)
    return V, V_mu


# Hyperparameters
chain_len = 4
horizon = chain_len  # Needs slightly more steps than chain len to reach end
learning_rates = [0.1, 0.5, 1., 2., 1e-3, 1e-5]
# learning_rates = [2.]
num_episodes = 100000
n_runs = 30

# Initialization
env = ChainEnv(chain_length=chain_len, horizon=horizon)
num_states = env.num_states
num_actions = env.num_actions

optimal_value, optimal_value_over_init_state = calculate_optimal_value_start(env)
print(f"Optimal Value V*(s=0, h=0) = {optimal_value[0][0]}")
print(f'Optimal Value V*(mu, h=0) = {optimal_value_over_init_state}')
print("-" * 30)

results_by_lr = {lr: {
    "all_runs_suboptimalities": np.zeros((n_runs, num_episodes)),
    "all_runs_suboptimalities_over_init_state": np.zeros((n_runs, num_episodes)),
    # "all_v_histories": [],
    # "all_prob_action_1_histories": [],
    "all_v_histories": np.zeros((n_runs, num_episodes, horizon, num_states)),
    "all_prob_action_1_histories": np.zeros((n_runs, num_episodes, horizon, num_states)),
} for lr in learning_rates}

for run in range(n_runs):
    print(f"\n==== Run {run+1} ===")

    for lr in learning_rates:
        print(f"Training with LR={lr}")
        # --- Policy Parameters ---
        # A list where each element is the parameter table (logits) for that time step
        policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
        suboptimality_history = []
        suboptimality_history_over_init_state = []
        v_history = []
        prob_action_1_history = []

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

            value_function, value_function_over_init_state = calculate_value_start(env, policy_params)
            suboptimality = optimal_value[0][0] - value_function[0][0]
            suboptimality_over_init_state = optimal_value_over_init_state - value_function_over_init_state
            suboptimality_history.append(suboptimality)
            suboptimality_history_over_init_state.append(suboptimality_over_init_state)
            v_snapshot = np.zeros((horizon, num_states))
            prob_action_1_snapshot = np.zeros((horizon, num_states))
            for h in range(horizon):
                for s in range(num_states):
                    v_snapshot[h, s] = value_function[h][s]

                    probs = softmax(policy_params[h][s, :])
                    prob_action_1_snapshot[h, s] = probs[1]
            v_history.append(v_snapshot)
            prob_action_1_history.append(prob_action_1_snapshot)

        results_by_lr[lr]["all_runs_suboptimalities"][run] = suboptimality_history
        results_by_lr[lr]["all_runs_suboptimalities_over_init_state"][run] = suboptimality_history_over_init_state
        # results_by_lr[lr]["all_v_histories"].append(np.array(v_history))
        # results_by_lr[lr]["all_prob_action_1_histories"].append(np.array(prob_action_1_history))
        results_by_lr[lr]["all_v_histories"][run] = np.array(v_history)
        results_by_lr[lr]["all_prob_action_1_histories"][run] = np.array(prob_action_1_history)

        np.save(f'./chain/all_runs_sub_lr_{lr}_run_{run+18}.npy', suboptimality_history)
        np.save(f'./chain/all_runs_sub_run_over_init_state_lr_{lr}_run_{run+18}.npy', suboptimality_history_over_init_state)
        np.save(f'./chain/all_v_histories_lr_{lr}_run_{run+18}.npy', np.array(v_history))
        np.save(f'./chain/all_prob_action_1_histories_lr_{lr}_run_{run+18}.npy', np.array(prob_action_1_history))

for lr in learning_rates:
    all_runs_suboptimalities = results_by_lr[lr]["all_runs_suboptimalities"]
    all_runs_suboptimalities_over_init_state = results_by_lr[lr]["all_runs_suboptimalities_over_init_state"]
    all_v_histories = np.array(results_by_lr[lr]["all_v_histories"])
    all_prob_action_1_histories = np.array(results_by_lr[lr]["all_prob_action_1_histories"])

    results_by_lr[lr]["avg_suboptimality"] = np.mean(all_runs_suboptimalities, axis=0)
    results_by_lr[lr]["std_suboptimality"] = np.std(all_runs_suboptimalities, axis=0)
    results_by_lr[lr]["avg_suboptimality_over_init_state"] = np.mean(all_runs_suboptimalities_over_init_state,
                                                                     axis=0)
    results_by_lr[lr]["std_suboptimality_over_init_state"] = np.std(all_runs_suboptimalities_over_init_state,
                                                                    axis=0)
    results_by_lr[lr]["mean_v"] = np.mean(all_v_histories, axis=0)
    results_by_lr[lr]["std_v"] = np.std(all_v_histories, axis=0)
    results_by_lr[lr]["mean_prob_action_1"] = np.mean(all_prob_action_1_histories, axis=0)
    results_by_lr[lr]["std_prob_action_1"] = np.std(all_prob_action_1_histories, axis=0)
    results_by_lr[lr]["episodes"] = np.arange(1, num_episodes + 1)

np.save('./chain/training_results_chain_mdp.npy', results_by_lr)

episodes = np.arange(1, num_episodes + 1)
# Average suboptimality for a specific run
for n in range(n_runs):
    plt.figure(figsize=(10, 6))
    for lr, data in results_by_lr.items():
        plt.plot(episodes, data["all_runs_suboptimalities"][n, :], label=f"$\\eta$={lr}", linewidth=1.5)

    plt.xlabel("Episodes (t)")
    plt.ylabel(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
    plt.title(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) in the episode {n+1}")
    plt.grid(True)
    plt.yscale('log')
    plt.legend(title="Learning Rate")
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./chain/suboptimality_chain_mdp_run_{n}_{timestamp}.png")
    # plt.show()
    plt.close()

print("-" * 30)
print("Training finished.")

for n in range(n_runs):
    fixed_episodes = 5000
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, lr in enumerate(learning_rates[:4]):
        ax = axes[idx]
        optimal_prob = np.array(results_by_lr[lr]["all_prob_action_1_histories"][n])[:fixed_episodes]  # [episodes, horizon, states]
        episodes = np.arange(fixed_episodes)

        horizon = optimal_prob.shape[1]  # Assuming [episodes, horizon, states]

        for h in range(horizon):
            ax.plot(episodes, optimal_prob[:, h, h], label=f"$\pi_T^{{{h}}}(a_1)$")

        ax.set_title(f"$\\eta$ = {lr}")
        ax.set_xlabel("Episodes (t)")
        ax.set_ylabel("Optimal policy ($\pi_T^h(a_1)$)")
        ax.grid(True)
        ax.legend(fontsize="small")

    plt.suptitle(f"Optimal policy ($\pi_T^h(a_1)$) evolution over first {fixed_episodes} episodes of run {n}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./chain/optimal_policy_evolution_over_first_{fixed_episodes}_episodes_{n}_run_{timestamp}.png")
    plt.close()


# Average suboptimality over all runs for specific learning rates
plt.figure(figsize=(10,6))
specific_lr = [1e-5, 1e-3, 0.1, 0.5]
for lr in specific_lr:
    episodes = results_by_lr[lr]["episodes"]
    mean_vals = results_by_lr[lr]["avg_suboptimality_over_init_state"]
    std_vals = results_by_lr[lr]["std_suboptimality_over_init_state"]

    plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
    plt.plot(episodes, mean_vals, label=f"$\\eta$={lr}", linewidth=1.5)

plt.xlabel("Episodes (t) ")
plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
plt.title(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) over {n_runs} runs with $\\eta =[0.1, 0.00001]$")
plt.grid(True)
plt.yscale('log')
plt.minorticks_on()
plt.legend(title="Learning Rate")
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./chain/suboptimality_chain_mdp_specific_lr_{timestamp}.png")
# plt.show()
plt.close()

# Average suboptimality over all runs
plt.figure(figsize=(10,6))
for lr, data in results_by_lr.items():
    episodes = data["episodes"]
    mean_vals = data["avg_suboptimality_over_init_state"]
    std_vals = data["std_suboptimality_over_init_state"]

    plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
    plt.plot(episodes, mean_vals, label=f"LR={lr}", linewidth=1.5)

plt.xlabel("Episodes (t)")
plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
plt.title(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) over {n_runs} runs")
plt.grid(True)
plt.yscale('log')
plt.minorticks_on()
plt.legend(title="Learning Rate")
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./chain/suboptimality_chain_mdp_{timestamp}.png")
# plt.show()
plt.close()

# Value function evolution over episodes
fixed_episodes = 1000
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates[:4]):
    ax = axes[idx]
    mean_v_values = np.array(results_by_lr[lr]["mean_v"])[:fixed_episodes]  # [episodes, horizon, states]
    std_v_values = np.array(results_by_lr[lr]["std_v"])[:fixed_episodes]
    episodes = np.arange(fixed_episodes)

    horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]

    for h in range(horizon):
        ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h], mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
        ax.plot(episodes, mean_v_values[:, h, h], label=f"$V^{{\\pi_T}}_{{{h}}}({h})$")

    ax.set_title(f"$\\eta$ = {lr}")
    ax.set_xlabel("Episodes (t)")
    ax.set_ylabel("Value function ($V^{\\pi_T}_h(s)$)")
    ax.grid(True)
    ax.legend(fontsize="small")

plt.suptitle(f"Value function ($V^{{\\pi_T}}_h(s)$) evolution over first {fixed_episodes} episodes", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./chain/value_function_evolution_over_first_{fixed_episodes}_episodes_{timestamp}.png")
plt.close()

# Optimal policy evolution over episodes
fixed_episodes = 5000
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates[:4]):
    ax = axes[idx]
    mean_v_values = np.array(results_by_lr[lr]["mean_prob_action_1"])[:fixed_episodes]  # [episodes, horizon, states]
    std_v_values = np.array(results_by_lr[lr]["std_prob_action_1"])[:fixed_episodes]
    episodes = np.arange(fixed_episodes)

    horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]

    for h in range(horizon):
        ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h], mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
        ax.plot(episodes, mean_v_values[:, h, h], label=f"$\pi_T^{{{h}}}(a_1)$")

    ax.set_title(f"$\\eta$ = {lr}")
    ax.set_xlabel("Episodes (t)")
    ax.set_ylabel("Optimal policy ($\pi_T^h(a_1)$)")
    ax.grid(True)
    ax.legend(fontsize="small")

plt.suptitle(f"Optimal policy ($\pi_T^h(a_1)$) evolution over first {fixed_episodes} episodes", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./chain/optimal_policy_evolution_over_first_{fixed_episodes}_episodes_{timestamp}.png")
plt.close()