import numpy as np
import os
from datetime import datetime
from environment.cart_pole_mdp import CartPoleDiscreteEnv
from utils.softmax import softmax
from utils.plot import plot_average_subopt
import matplotlib.pyplot as plt


def get_action(state, step_h, env, policy_params, num_actions):
    state_idx = env._state_to_idx(state)
    logits = policy_params[step_h][state_idx]
    probs = softmax(logits)
    action = np.random.choice(num_actions, p=probs)
    return action, probs


def main(env, horizon, learning_rates, num_episodes, num_runs, title="cartpole_pg", is_show_plot=False):
    timestamp = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    dir_path = f"./exp/{title}/{timestamp}/"
    os.makedirs(dir_path, exist_ok=True)

    num_states = env.num_states
    num_actions = env.num_actions

    results_by_lr = {lr: {
        "returns": np.zeros((num_runs, num_episodes)),
    } for lr in learning_rates}

    for run in range(num_runs):
        print(f"==== Run {run + 1} ===")

        for lr in learning_rates:
            print(f"Training with LR={lr}")
            policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
            returns_history = []

            for episode in range(num_episodes):
                state = env.reset()
                episode_data = []

                for h in range(horizon):
                    action, probs = get_action(state, h, env, policy_params, num_actions)
                    next_state, reward, done = env.step(action)
                    episode_data.append({"state": state, "action": action, "reward": reward, "probs": probs})
                    state = next_state
                    if done:
                        break

                # Compute returns-to-go
                Gs = np.zeros(len(episode_data))
                G = 0
                for h in reversed(range(len(episode_data))):
                    G = episode_data[h]["reward"] + G
                    Gs[h] = G

                # Policy gradient update
                for h in range(len(episode_data)):
                    s = episode_data[h]["state"]
                    a = episode_data[h]["action"]
                    G_h = Gs[h]
                    probs = episode_data[h]["probs"]
                    s_idx = env._state_to_idx(s)

                    grad_log_pi = -probs
                    grad_log_pi[a] += 1.0
                    policy_params[h][s_idx] += lr * G_h * grad_log_pi

                returns_history.append(np.sum([step["reward"] for step in episode_data]))

            results_by_lr[lr]["returns"][run] = np.array(returns_history)

    print("Training complete.")

    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        mean_rewards = np.mean(results_by_lr[lr]["returns"], axis=0)
        std_rewards = np.std(results_by_lr[lr]["returns"], axis=0)
        episodes = np.arange(1, len(mean_rewards) + 1)
        plt.plot(episodes, mean_rewards, label=f"LR={lr:.0e}")
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Average Episode Return")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    os.makedirs(dir_path, exist_ok=True)
    plot_file = os.path.join(dir_path, f"{title}.png")
    plt.savefig(plot_file)
    show_plot = False
    if show_plot:
        plt.show()
    plt.close()

    return results_by_lr

if __name__ == "__main__":
    env = CartPoleDiscreteEnv(bins=(6, 6, 6, 6), horizon=200)
    learning_rates = [0.00001, 0.0001, 0.01, 1.]
    results = main(env, horizon=200, learning_rates=learning_rates,
                   num_episodes=10000, num_runs=5, is_show_plot=False)
