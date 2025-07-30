import matplotlib.pyplot as plt
import numpy as np

def plot_supopt_each_run(num_runs, results_by_lr, plot_path, episodes, show_plot=False):
    # Average suboptimality for a specific run
    for n in range(num_runs):
        plt.figure(figsize=(10, 6))
        for lr, data in results_by_lr.items():
            plt.plot(episodes, data["suboptimality_over_init_states_histories"][n, :], label=f"$\\eta$={lr}", linewidth=1.5)

        plt.xlabel("Episodes (t)")
        plt.ylabel(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$)")
        plt.title(f"Suboptimality $(V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$) in the episode {n + 1}")
        plt.grid(True)
        plt.yscale('log')
        plt.legend(title="Learning Rate")
        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.savefig(f"{plot_path}suboptimality_run_{n}.png")
        plt.close()

def plot_average_subopt(num_runs, results_by_lr, learning_rate, title, plot_path, episodes, show_plot=False):
    # Average suboptimality over all runs for specific learning rates
    plt.figure(figsize=(10, 6))
    for lr in learning_rate:
        mean_vals = np.mean(results_by_lr[lr]["suboptimality_over_init_states_histories"], axis=0)
        std_vals = np.std(results_by_lr[lr]["suboptimality_over_init_states_histories"], axis=0)

        plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
        plt.plot(episodes, mean_vals, label=f"$\\eta$={lr}", linewidth=1.5)

    plt.xlabel("Episodes (t) ")
    plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$)")
    plt.title(
        f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$) over {num_runs} runs")
    plt.grid(True)
    plt.yscale('log')
    plt.minorticks_on()
    plt.legend(title="Learning Rate")
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_path}{title}.png")
    plt.close()

def plot_average_supopt_last_iterate(num_runs, results_by_lr, learning_rates, episodes, title, plot_path, is_show_plot=False):
    plt.figure(figsize=(10, 6))
    average_subopt_last_iterate = []
    std_subopt_last_iterate = []
    for lr in learning_rates:
        mean_vals = np.mean(results_by_lr[lr]["suboptimality_over_init_states_histories"][-1], axis=0)
        std_vals = np.std(results_by_lr[lr]["suboptimality_over_init_states_histories"][-1], axis=0)
        average_subopt_last_iterate.append(mean_vals)
        std_subopt_last_iterate.append(std_vals)
        # std_vals = np.std(results_by_lr[lr]["suboptimality_over_init_states_histories"], axis=0)
        #
        # plt.fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
        # plt.plot(episodes, mean_vals, label=f"$\\eta$={lr}", linewidth=1.5)

    plt.errorbar(learning_rates, np.array(average_subopt_last_iterate), yerr=np.array(std_subopt_last_iterate), linestyle='None', marker='o', capsize=5, label="Average Suboptimality")
    plt.xlabel(f"Learning rates ($\eta$) ")
    plt.ylabel(f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$)")
    plt.title(
        f"Average suboptimality ($V^*_0(\\rho) - V^{{\\pi_t}}_0(\\rho)$) of {len(learning_rates)} learning rates")
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_on()
    # plt.legend(title="Learning Rate")
    plt.tight_layout()
    if is_show_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_path}{title}.png")
    plt.close()

# Optimal policy evolution over episodes
def plot_opt_policy(results_by_lr, fixed_episodes, learning_rates, plot_path, show_plot=False):
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(18, 5))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        mean_v_values = np.array(np.mean(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"], axis=0))[
                        :fixed_episodes]  # [episodes, horizon, states]
        std_v_values = np.array(np.std(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"], axis=0))[:fixed_episodes]
        episodes = np.arange(fixed_episodes)

        horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]

        for h in range(horizon):
            ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h],
                            mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
            ax.plot(episodes, mean_v_values[:, h, h], label=f"$\pi_t^{{{h}}}(a_1)$")

        ax.set_title(f"$\\eta$ = {lr}")
        # ax.set_xlabel("Episodes (t)")
        # ax.set_ylabel("Probability assigned to optimal action ($\pi_t^h(a_1)$)")
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize="small")

    fig.supxlabel("Episodes (t)", y=0.05)
    fig.supylabel("Probability assigned to optimal actions", x=0.0)
    plt.suptitle(f"Probability assigned to optimal actions ($\pi_t^h(a_1)$) in first {fixed_episodes} episodes",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if show_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_path}optimal_policy_in_first_{fixed_episodes}_episodes.png")
    plt.close()


def plot_opt_policy_each_run(num_runs, results_by_lr, fixed_episodes, learning_rates, plot_path, show_plot=False):
    for n in range(num_runs):
        fig, axes = plt.subplots(1, len(learning_rates[:4]), figsize=(18, 5))
        axes = axes.flatten()
        for idx, lr in enumerate(learning_rates):
            ax = axes[idx]
            optimal_prob = np.array(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"][n])[
                           :fixed_episodes]  # [episodes, horizon, states]
            episodes = np.arange(fixed_episodes)

            horizon = optimal_prob.shape[1]  # Assuming [episodes, horizon, states]

            for h in range(horizon):
                ax.plot(episodes, optimal_prob[:, h, h], label=f"$\pi_t^{{{h}}}(a_1)$")

            ax.set_title(f"$\\eta$ = {lr}")
            # ax.set_xlabel("Episodes (t)")
            # ax.set_ylabel("Probability assigned to optimal actions ($\pi_t^h(a_1|s)$)")
            ax.grid(True)
            if idx == 0:
                ax.legend(fontsize="small")

        fig.supxlabel("Episodes (t)", y=0.05)
        fig.supylabel("Probability assigned to optimal actions", x=0.01)
        plt.suptitle(
            f"Probability assigned to optimal actions ($\pi_t^h(a_1)$) in first {fixed_episodes} episodes of run {n}",
            fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        if show_plot:
            plt.show()
        else:
            plt.savefig(f"{plot_path}optimal_policy_in_first_{fixed_episodes}_episodes_run_{n}.png")
        plt.close()


def plot_opt_policy_tree_mdp(results_by_lr, fixed_episodes, learning_rates, plot_path, show_plot=False):
    if is_multiple_optimal:
        prob_0 = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        prob_1 = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        total_prob = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        for idx, lr in enumerate(learning_rates[:4]):
            for n in range(num_runs):
                prob_0[idx, n, :] = results_by_lr[lr]["prob_histories"][n, :, 0, 0, 0]
                prob_1[idx, n, :] = results_by_lr[lr]["prob_histories"][n, :, 0, 0, 1]
                total_prob[idx, n, :] = prob_0[idx, n, :] + prob_1[idx, n, :]

    fig, axes = plt.subplots(1, len(learning_rates), figsize=(18, 5))
    axes = axes.flatten()

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        mean_v_values = np.array(np.mean(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"], axis=0))[
                        :fixed_episodes]  # [episodes, horizon, states]
        std_v_values = np.array(np.std(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"], axis=0))[:fixed_episodes]
        episodes = np.arange(fixed_episodes)

        horizon = mean_v_values.shape[1]  # Assuming [episodes, horizon, states]

        for h in range(horizon):
            ax.fill_between(episodes, mean_v_values[:, h, h] - std_v_values[:, h, h],
                            mean_v_values[:, h, h] + std_v_values[:, h, h], alpha=0.2)
            ax.plot(episodes, mean_v_values[:, h, h], label=f"$\pi_t^{{{h}}}(a_1)$")

        ax.set_title(f"$\\eta$ = {lr}")
        # ax.set_xlabel("Episodes (t)")
        # ax.set_ylabel("Probability assigned to optimal action ($\pi_t^h(a_1)$)")
        ax.grid(True)
        if idx == 0:
            ax.legend(fontsize="small")

    fig.supxlabel("Episodes (t)", y=0.05)
    fig.supylabel("Probability assigned to optimal actions", x=0.0)
    plt.suptitle(f"Probability assigned to optimal actions ($\pi_t^h(a_1)$) in first {fixed_episodes} episodes",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    if show_plot:
        plt.show()
    else:
        plt.savefig(f"{plot_path}optimal_policy_in_first_{fixed_episodes}_episodes.png")
    plt.close()


def plot_opt_policy_each_run_tree_mdp(num_runs, results_by_lr, fixed_episodes, learning_rates, plot_path, show_plot=False):
    for n in range(num_runs):
        fig, axes = plt.subplots(1, len(learning_rates[:4]), figsize=(18, 5))
        axes = axes.flatten()
        for idx, lr in enumerate(learning_rates):
            ax = axes[idx]
            optimal_prob = np.array(results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"][n])[
                           :fixed_episodes]  # [episodes, horizon, states]
            episodes = np.arange(fixed_episodes)

            horizon = optimal_prob.shape[1]  # Assuming [episodes, horizon, states]

            for h in range(horizon):
                ax.plot(episodes, optimal_prob[:, h, h], label=f"$\pi_t^{{{h}}}(a_1)$")

            ax.set_title(f"$\\eta$ = {lr}")
            # ax.set_xlabel("Episodes (t)")
            # ax.set_ylabel("Probability assigned to optimal actions ($\pi_t^h(a_1|s)$)")
            ax.grid(True)
            if idx == 0:
                ax.legend(fontsize="small")

        fig.supxlabel("Episodes (t)", y=0.05)
        fig.supylabel("Probability assigned to optimal actions", x=0.01)
        plt.suptitle(
            f"Probability assigned to optimal actions ($\pi_t^h(a_1)$) in first {fixed_episodes} episodes of run {n}",
            fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        if show_plot:
            plt.show()
        else:
            plt.savefig(f"{plot_path}optimal_policy_in_first_{fixed_episodes}_episodes_run_{n}.png")
        plt.close()

