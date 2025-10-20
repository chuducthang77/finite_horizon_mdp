import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./bandit/data.pkl', 'rb') as f:
    results = pickle.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.flatten()

axes[0].plot(results[1]['pi1'][9, :], label=r"$\pi_{\theta_t}(a_1)$")
axes[0].plot(results[1]['pi2'][9, :], label=r"$\pi_{\theta_t}(a_2)$")
axes[0].plot(results[1]['pi_optimal_total'][9, :], label=r"$\pi_{\theta_t}(a_1) + \pi_{\theta_t}(a_2)$")
axes[0].set_ylabel('Probability assigned to optimal arm')
# axes[0].set_xlabel('Episodes (t)')
axes[0].grid(True)
axes[0].set_title(r"$\eta=1$")
axes[0].legend()


axes[1].plot(results[10]['pi1'][3, :], label=r"$\pi_{\theta_t}(a_1)$")
axes[1].plot(results[10]['pi2'][3, :], label=r"$\pi_{\theta_t}(a_2)$")
axes[1].plot(results[10]['pi_optimal_total'][3, :], label=r"$\pi_{\theta_t}(a_1) + \pi_{\theta_t}(a_2)$")
# axes[1].set_ylabel('Probability assigned to optimal arm')
# axes[1].set_xlabel('Episodes (t)')
axes[1].grid(True)
axes[1].set_title(r"$\eta=10$")
# axes[1].legend()

# plt.suptitle(r"Total probability assigned to optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*)$", fontsize=12, y=0.9)
fig.supxlabel('Episodes (t)', y =0.12)
plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 
plt.savefig("./bandit/large_learning_rate_total_prob_astar_eta_.pdf", dpi=3000)