import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

with open('./tree/training_results_tree_mdp_probability_action_1_20250513_193009.npy', 'rb') as f:
    prob_0 = np.load(f, allow_pickle=True)

with open('./tree/training_results_tree_mdp_probability_action_2_20250513_193009.npy', 'rb') as f:
    prob_1 = np.load(f, allow_pickle=True)

with open('./tree/training_results_tree_mdp_total_probability_20250513_193009.npy', 'rb') as f:
    total_prob = np.load(f, allow_pickle=True)

num_actions = 3
horizon = 2
learning_rates = [0.1, 0.5, 1., 2., 1e-3, 1e-5]
num_episodes = 100000
n_runs = 10

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

plt.suptitle(r"Total probability assigned to optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*|s_1)$")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"./tree/total_prob_astar_tree_mdp_over_{n_runs}_runs_{timestamp}_.png")

for n in range(n_runs):
    plt.figure(figsize=(10, 6))
    for idx, eta in enumerate(learning_rates[:4]):
        plt.plot(prob_0[idx, n, :], label=r"$\pi_{\theta_t}(a_1|s_1)$")
        plt.plot(prob_1[idx, n, :], label=r"$\pi_{\theta_t}(a_2|s_1)$")
        plt.plot(total_prob[idx, n, :], label=r"$\pi_{\theta_t}(a_1|s_1) + \pi_{\theta_t}(a_2|s_1)$")
        plt.ylabel('Probability of optimal arms')
        plt.xlabel('Episodes (t)')
        plt.grid(True)
        plt.title(r"Total probability assigned to optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*)$")
        plt.legend()
        plt.savefig('./tree/large_learning_rate_total_prob_astar_eta_{eta:.2f}_run_{n}.pdf'.format(eta=eta, n=n), dpi=3000)
        plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

axes[0].plot(prob_0[0, 4, :], label=r"$\pi_{\theta_t}(a_1|s_1)$")
axes[0].plot(prob_1[0, 4, :], label=r"$\pi_{\theta_t}(a_2|s_1)$")
axes[0].plot(total_prob[0, 4, :], label=r"$\pi_{\theta_t}(a_1|s_1) + \pi_{\theta_t}(a_2|s_1)$")
axes[0].set_ylabel('Probability assigned to optimal arms')
# axes[0].set_xlabel('Episodes (t)')
axes[0].grid(True)
axes[0].set_title(r"$\eta=0.1$")
axes[0].legend()
axes[1].plot(prob_0[1, 7, :], label=r"$\pi_{\theta_t}(a_1|s_1)$")
axes[1].plot(prob_1[1, 7, :], label=r"$\pi_{\theta_t}(a_2|s_1)$")
axes[1].plot(total_prob[1, 7, :], label=r"$\pi_{\theta_t}(a_1|s_1) + \pi_{\theta_t}(a_2|s_1)$")
# axes[1].set_ylabel('Probability of optimal arms')
# axes[1].set_xlabel('Episodes (t)')
axes[1].grid(True)
axes[1].set_title(r"$\eta=0.5$")
# axes[1].legend()
# plt.suptitle(r"Total probability assigned to optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*|s_1)$")
fig.supxlabel('Episodes (t)', y =0.12)
plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 
plt.savefig(f"./tree/large_learning_rate_total_prob_astar_eta_0.1_0.5_{timestamp}_.png")
