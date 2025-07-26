import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from env.tree_mdp import TreeMDP

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
num_episodes = 100000
n_runs = 10

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

# np.save('./tree/training_results_tree_mdp.npy', results_by_lr)

episodes = np.arange(1, num_episodes + 1)
# Total probability of optimal arms
# Uncomment only when set (0,0): (1,1) in transitions. Otherwise, comment
prob_0 = np.zeros((len(learning_rates[:4]), n_runs, num_episodes))
prob_1 = np.zeros((len(learning_rates[:4]), n_runs, num_episodes))
total_prob = np.zeros((len(learning_rates[:4]), n_runs, num_episodes))
for idx, lr in enumerate(learning_rates[:4]):
    for n in range(n_runs):
        prob_0[idx, n, :] = results_by_lr[lr]["all_prob_action_histories"][n, :, 0, 0, 0]
        prob_1[idx, n, :] = results_by_lr[lr]["all_prob_action_histories"][n, :, 0, 0, 1]
        total_prob[idx, n, :] = prob_0[idx, n, :] + prob_1[idx, n, :]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
np.save(f'./tree/training_results_tree_mdp_probability_action_1_{timestamp}.npy', prob_0)
np.save(f'./tree/training_results_tree_mdp_probability_action_2_{timestamp}.npy', prob_1)
np.save(f'./tree/training_results_tree_mdp_total_probability_{timestamp}.npy', total_prob)

average_prob_0 = np.mean(prob_0, axis=1)
average_prob_1 = np.mean(prob_1, axis=1)
average_total_prob = np.mean(total_prob, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes = axes.flatten()
for idx, eta in enumerate(learning_rates[:4]):
    ax = axes[idx]
    ax.plot(average_prob_0[idx, :], label=r"$\pi_{\theta_t}(a_1|s_1)$")
    ax.plot(average_prob_1[idx, :], label=r"$\pi_{\theta_t}(a_2|s_1)$")
    ax.plot(average_total_prob[idx, :], label=r"$\pi_{\theta_t}(a_1|s_1) + \pi_{\theta_t}(a_2|s_1)$")
    ax.set_ylabel('Probability of optimal arms')
    ax.set_xlabel('Episodes (t)')
    ax.grid(True)
    ax.set_title(f"$\\eta = {eta}$")
    ax.legend()

plt.suptitle(r"Total probability of optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*|s_1)$")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./tree/total_prob_astar_tree_mdp_over_{n_runs}_runs_{timestamp}.png")

for n in range(n_runs):
    plt.figure(figsize=(10, 6))
    for idx, eta in enumerate(learning_rates[:4]):
        plt.plot(prob_0[idx, n, :], label=r"$\pi_{\theta_t}(a_1|s_1)$")
        plt.plot(prob_1[idx, n, :], label=r"$\pi_{\theta_t}(a_2|s_1)$")
        plt.plot(total_prob[idx, n, :], label=r"$\pi_{\theta_t}(a_1|s_1) + \pi_{\theta_t}(a_2|s_1)$")
        plt.ylabel('Probability of optimal arms')
        plt.xlabel('Episodes (t)')
        plt.grid(True)
        plt.title(r"Total probability of optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*)$")
        plt.legend()
        plt.savefig('./tree/large_learning_rate_total_prob_astar_eta_{eta:.2f}_run_{n}.pdf'.format(eta=eta, n=n), dpi=3000)
        plt.close()

# Average suboptimality for a specific run
# for n in range(n_runs):
#     plt.figure(figsize=(10, 6))
#     for lr, data in results_by_lr.items():
#         plt.plot(episodes, data["all_runs_suboptimalities"][n, :], label=f"$\\eta$={lr}", linewidth=1.5)
#
#     plt.xlabel("Episodes (t)")
#     plt.ylabel(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
#     plt.title(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) in the episode {n+1}")
#     plt.grid(True)
#     plt.yscale('log')
#     plt.legend(title="Learning Rate")
#     plt.tight_layout()
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f"suboptimality_tree_mdp_run_{n}_{timestamp}.png")
#     # plt.show()
#
#
# for n in range(n_runs):
#     fixed_episodes = 1000
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     axes = axes.flatten()
#     for idx, lr in enumerate(learning_rates[:4]):
#         ax = axes[idx]
#         optimal_prob = np.array(results_by_lr[lr]["all_prob_action_histories"][n])[:fixed_episodes]  # [episodes, horizon, states]
#         episodes = np.arange(fixed_episodes)
#
#         horizon = optimal_prob.shape[1]  # Assuming [episodes, horizon, states]
#
#         for h in range(horizon):
#             ax.plot(episodes, optimal_prob[:, h, h, 0], label=f"$\pi_T^{{{h}}}(0|{{{h}}})$")
#
#         ax.set_title(f"$\\eta$ = {lr}")
#         ax.set_xlabel("Episodes (t)")
#         ax.set_ylabel("Optimal policy ($\pi_T^h(a|s)$)")
#         ax.grid(True)
#         ax.legend(fontsize="small")
#
#     plt.suptitle(f"Optimal policy ($\pi_T^h(a_1)$) evolution over first {fixed_episodes} episodes of run {n}", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f"optimal_policy_evolution_over_first_{fixed_episodes}_episodes_{n}_run_{timestamp}.png")
#     plt.close()

#
# # Average suboptimality over all runs for specific learning rates
# plt.figure(figsize=(10,6))
# specific_lr = [1e-5, 1e-3, 0.1, 0.5]
# for lr in specific_lr:
#     episodes = results_by_lr[lr]["episodes"]
#     mean_vals = results_by_lr[lr]["avg_suboptimality_over_init_state"]
#     std_vals = results_by_lr[lr]["std_suboptimality_over_init_state"]
#
#     plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
#     plt.plot(episodes, mean_vals, label=f"$\\eta$={lr}", linewidth=1.5)
#
# plt.xlabel("Episodes (t) ")
# plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
# plt.title(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) over {n_runs} runs with $\\eta =[0.1, 0.00001]$")
# plt.grid(True)
# plt.yscale('log')
# plt.minorticks_on()
# plt.legend(title="Learning Rate")
# plt.tight_layout()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"suboptimality_tree_mdp_specific_lr_{timestamp}.png")
# # plt.show()
#
# # Average suboptimality over all runs
# plt.figure(figsize=(10,6))
# for lr, data in results_by_lr.items():
#     episodes = data["episodes"]
#     mean_vals = data["avg_suboptimality_over_init_state"]
#     std_vals = data["std_suboptimality_over_init_state"]
#
#     plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
#     plt.plot(episodes, mean_vals, label=f"LR={lr}", linewidth=1.5)
#
# plt.xlabel("Episodes (t)")
# plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
# plt.title(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) over {n_runs} runs")
# plt.grid(True)
# plt.yscale('log')
# plt.minorticks_on()
# plt.legend(title="Learning Rate")
# plt.tight_layout()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"suboptimality_tree_mdp_{timestamp}.png")
# # plt.show()
#
# # Value function evolution over episodes
# fixed_episodes = 1000
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes = axes.flatten()
#
# for idx, lr in enumerate(learning_rates[:4]):
#     ax = axes[idx]
#     mean_v_values = np.array(results_by_lr[lr]["mean_v"])[:fixed_episodes]  # [episodes, horizon, states]
#     std_v_values = np.array(results_by_lr[lr]["std_v"])[:fixed_episodes]
#     episodes = np.arange(fixed_episodes)
#
#     horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]
#
#     for h in range(horizon):
#         ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h], mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
#         ax.plot(episodes, mean_v_values[:, h, h], label=f"$V^{{\\pi_T}}_{{{h}}}({h})$")
#
#     ax.set_title(f"$\\eta$ = {lr}")
#     ax.set_xlabel("Episodes (t)")
#     ax.set_ylabel("Value function ($V^{\\pi_T}_h(s)$)")
#     ax.grid(True)
#     ax.legend(fontsize="small")
#
# plt.suptitle(f"Value function ($V^{{\\pi_T}}_h(s)$) evolution over first {fixed_episodes} episodes", fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"value_function_evolution_over_first_{fixed_episodes}_episodes_{timestamp}.png")
#
# # Optimal policy evolution over episodes
# fixed_episodes = 5000
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes = axes.flatten()
#
# for idx, lr in enumerate(learning_rates[:4]):
#     ax = axes[idx]
#     mean_v_values = np.array(results_by_lr[lr]["mean_prob_action_1"])[:fixed_episodes]  # [episodes, horizon, states]
#     std_v_values = np.array(results_by_lr[lr]["std_prob_action_1"])[:fixed_episodes]
#     episodes = np.arange(fixed_episodes)
#
#     horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]
#
#     for h in range(horizon):
#         ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h], mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
#         ax.plot(episodes, mean_v_values[:, h, h], label=f"$\pi_T^{{{h}}}(a_1)$")
#
#     ax.set_title(f"$\\eta$ = {lr}")
#     ax.set_xlabel("Episodes (t)")
#     ax.set_ylabel("Optimal policy ($\pi_T^h(a_1)$)")
#     ax.grid(True)
#     ax.legend(fontsize="small")
#
# plt.suptitle(f"Optimal policy ($\pi_T^h(a_1)$) evolution over first {fixed_episodes} episodes", fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# plt.savefig(f"optimal_policy_evolution_over_first_{fixed_episodes}_episodes_{timestamp}.png")