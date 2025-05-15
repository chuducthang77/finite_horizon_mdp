import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# results_by_lr = np.load('training_results_chain_mdp.npy',allow_pickle='TRUE').item()

chain_len = 4
horizon = chain_len  # Needs slightly more steps than chain len to reach end
learning_rates = [0.1, 0.5, 1., 2., 0.001, 0.00001]
# learning_rates = [2.]
num_episodes = 100000
n_runs = 30
num_states = chain_len + 1

episodes = np.arange(1, num_episodes + 1)
results_by_lr = {lr: {
    "all_runs_suboptimalities": np.zeros((n_runs, num_episodes)),
    "all_runs_suboptimalities_over_init_state": np.zeros((n_runs, num_episodes)),
    # "all_v_histories": [],
    # "all_prob_action_1_histories": [],
    "all_v_histories": np.zeros((n_runs, num_episodes, horizon, num_states)),
    "all_prob_action_1_histories": np.zeros((n_runs, num_episodes, horizon, num_states)),
} for lr in learning_rates}

titles = ["all_runs_sub_lr_", "all_runs_sub_run_over_init_state_lr_", "all_v_histories_lr_", "all_prob_action_1_histories_lr_"]
for title in titles:
    for lr in learning_rates:
        for n in range(n_runs):
            # Load the results for each run
            if lr == 1e-5:
                with open(f"./chain/{title}1e-05_run_{n}.npy", "rb") as f:
                    results = np.load(f)
            else:
                with open(f"./chain/{title}{lr}_run_{n}.npy", "rb") as f:
                    results = np.load(f)
            # Store the results in the dictionary
            if title == "all_runs_sub_lr_":
                results_by_lr[lr]["all_runs_suboptimalities"][n] = results
            elif title == "all_runs_sub_run_over_init_state_lr_":
                results_by_lr[lr]["all_runs_suboptimalities_over_init_state"][n] = results
            elif title == "all_v_histories_lr_":
                results_by_lr[lr]["all_v_histories"][n] = results
            elif title == "all_prob_action_1_histories_lr_":
                results_by_lr[lr]["all_prob_action_1_histories"][n] = results
        print('lr: ', lr)

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
# print("Training finished.")

for n in range(n_runs):
    fixed_episodes = 5000
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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
    plt.close(fig)

plt.figure(figsize=(10,6))
specific_lr = [1e-5, 1e-3, 0.1]
for lr in specific_lr:
    episodes = results_by_lr[lr]["episodes"]
    mean_vals = results_by_lr[lr]["avg_suboptimality_over_init_state"]
    std_vals = results_by_lr[lr]["std_suboptimality_over_init_state"]

    plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
    plt.plot(episodes, mean_vals, label=f"$\\eta$={lr}", linewidth=1.5)

plt.xlabel("Episodes (t) ")
plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$)")
plt.title(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_T}}_0(\\rho)$) over {n_runs} runs with $\\eta =[1e-5, 10e-3, 0.1]$")
plt.grid(True)
plt.yscale('log')
plt.minorticks_on()
plt.legend(title="Learning Rate")
plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./chain/suboptimality_chain_mdp_specific_lr_{timestamp}.png")
# plt.show()
plt.close()

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

fixed_episodes = 1000
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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
plt.close(fig)

fixed_episodes = 5000
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
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
plt.close(fig)