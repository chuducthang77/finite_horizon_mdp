import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import argparse
import os
import yaml

from env.tree_mdp import TreeEnv
from utils.softmax import softmax
from utils.calculate_v_tree_mdp import calculate_value_function
from utils.calculate_v_star_tree_mdp import calculate_optimal_value_function
from utils.plot import *


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


def main(env, horizon, learning_rates, num_episodes, num_runs, is_show_plot=False, is_save_ind_file=True,
         is_save_whole_file=True, is_multiple_optimal=True):
    # Saving the result
    timestamp = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    dir_path = f"./exp/{env}/{timestamp}/"
    plot_path = dir_path + "plots/"
    model_path = dir_path + "models/"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Hyperparameters
    env = TreeEnv()
    states = env.num_states - env.num_terminal_states

    num_states = len(states)
    num_actions = 3
    horizon = horizon
    learning_rates = learning_rates
    num_episodes = num_episodes
    episodes = np.arange(1, num_episodes + 1)
    num_runs = num_runs

    # Results storage
    results_by_lr = {lr: {
        "suboptimality_histories": np.zeros((num_runs, num_episodes)),
        "suboptimality_over_init_states_histories": np.zeros((num_runs, num_episodes)),
        "prob_histories": np.zeros((num_runs, num_episodes, horizon, num_states, num_actions)),
    } for lr in learning_rates}

    # Optimal value
    optimal_value, optimal_value_over_init_states = calculate_optimal_value_function(env, horizon)
    print(f"Optimal Value V*(s=0, h=0) = {optimal_value[0][0]}")
    print(f"Optimal value V*(mu, h=0) = {optimal_value_over_init_states}")
    print("-" * 30)

    for run in range(num_runs):
        print(f"\n==== Run {run + 1} ===")
        for lr in learning_rates:
            print(f"Training with LR={lr}")
            policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
            suboptimality_history = []
            suboptimality_history_over_init_state = []

            for episode in range(num_episodes):
                state = env.reset()
                episode_data = []
                total_reward = 0

                for h in range(horizon):
                    logits = policy_params[h][state]
                    probs = softmax(logits)
                    action = np.random.choice(num_actions, p=probs)

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

                # Compute returns
                returns_to_go = np.zeros(len(episode_data))
                cumulative_return = 0
                for h in reversed(range(len(episode_data))):
                    cumulative_return = episode_data[h]["reward"] + cumulative_return
                    returns_to_go[h] = cumulative_return

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
                value_function, value_function_over_init_state = calculate_value_tree(env, policy_params, horizon=horizon)
                suboptimality = optimal_value[0][0] - value_function[0][0]
                suboptimality_over_init_state = optimal_value_over_init_states - value_function_over_init_state

                suboptimality_history.append(suboptimality)
                suboptimality_history_over_init_state.append(suboptimality_over_init_state)

                for h in range(horizon):
                    for s_idx in range(num_states):
                        probs = softmax(policy_params[h][s_idx])
                        results_by_lr[lr]["prob_histories"][run, episode, h, s_idx] = probs

            results_by_lr[lr]["suboptimality_histories"][run] = np.array(suboptimality_history)
            results_by_lr[lr]["suboptimality_over_init_states_histories"][run] = np.array(suboptimality_history_over_init_state)

            if is_save_ind_file:
                np.save(f'{model_path}subopt_lr_{lr}_run_{run}.npy', np.array(suboptimality_history))
                np.save(f'{model_path}subopt_over_init_lr_{lr}_run_{run}.npy', np.array(suboptimality_history_over_init_state))
                np.save(f'{model_path}prob_lr_{lr}_run_{run}.npy', results_by_lr[lr]["prob_histories"][run])

    print("-" * 30)
    print("Training finished.")

    # Total probability of optimal arms
    # Uncomment only when set (0,0): (1,1) in transitions. Otherwise, comment
    if is_multiple_optimal:
        prob_0 = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        prob_1 = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        total_prob = np.zeros((len(learning_rates[:4]), num_runs, num_episodes))
        for idx, lr in enumerate(learning_rates[:4]):
            for n in range(num_runs):
                prob_0[idx, n, :] = results_by_lr[lr]["prob_histories"][n, :, 0, 0, 0]
                prob_1[idx, n, :] = results_by_lr[lr]["prob_histories"][n, :, 0, 0, 1]
                total_prob[idx, n, :] = prob_0[idx, n, :] + prob_1[idx, n, :]

        np.save(f'{model_path}training_results_tree_mdp_probability_action_1.npy', prob_0)
        np.save(f'{model_path}training_results_tree_mdp_probability_action_2.npy', prob_1)
        np.save(f'{model_path}training_results_tree_mdp_total_probability.npy', total_prob)

    if is_multiple_optimal:
        plot_opt_policy(results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
        plot_opt_policy_each_run(num_runs, results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
    else:
        plot_average_subopt(num_runs, results_by_lr, learning_rates, "average_suboptimality", plot_path, episodes,
                            is_show_plot)
        plot_average_subopt(num_runs, results_by_lr, [1e-5, 1e-3, 0.1], "average_suboptimality_specific_lr", plot_path,
                            episodes, is_show_plot)
        plot_supopt_each_run(num_runs, results_by_lr, plot_path, episodes, is_show_plot)
        plot_opt_policy(results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
        plot_opt_policy_each_run(num_runs, results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='ChainMDP')
    parser.add_argument('--chain_len', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--learning_rates', type=list, default=[0.1, 0.5, 1., 2., 1e-3, 1e-5])
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_runs', type=int, default=1)

    args = parser.parse_args()

    with open("./config/tree.yaml", "r") as f:
        config = yaml.safe_load(f)

    # main(env = args.env_name, chain_len = args.chain_len, horizon = args.horizon, learning_rates = args.learning_rates,
    #      num_episodes = args.num_episodes, num_runs = args.num_runs)

    main(env=config['env_name'], horizon=config['horizon'],
         learning_rates=list(config['learning_rates']),
         num_episodes=config['num_episodes'], num_runs=config['num_runs'], is_show_plot=config['is_show_plot'],
         is_save_ind_file=config['is_save_ind_file'], is_save_whole_file=config['is_save_whole_file'], is_multiple_optimal=config['is_multiple_optimal'])
