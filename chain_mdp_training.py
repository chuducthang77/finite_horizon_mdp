import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os
import yaml

from environment.chain_mdp import ChainEnv
from utils.softmax import softmax
from utils.calculate_v_chain_mdp import calculate_value_function
from utils.calculate_v_star_chain_mdp import calculate_optimal_value_function
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


def main(env, chain_len, horizon, learning_rates, num_episodes, num_runs, is_show_plot=False, is_save_plot=False, is_save_ind_file=True,
         is_save_whole_file=True):
    # Saving the result
    timestamp = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")
    dir_path = f"./exp/{env}/{timestamp}/"
    plot_path = dir_path + "plots/"
    model_path = dir_path + "models/"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Hyperparameters
    chain_len = chain_len
    horizon = chain_len  # Needs slightly more steps than chain len to reach end
    initial_learning_rates = len(learning_rates)
    if initial_learning_rates > 0:
        learning_rates = learning_rates
    else:
        learning_rates = np.linspace(-9, 0, num = 50)
        learning_rates = np.exp(learning_rates)
    num_episodes = num_episodes
    episodes = np.arange(1, num_episodes + 1)
    num_runs = num_runs

    # Initialization
    env = ChainEnv(chain_length=chain_len, horizon=horizon)
    num_states = env.num_states
    num_actions = env.num_actions

    results_by_lr = {lr: {
        "suboptimality_histories": np.zeros((num_runs, num_episodes)),
        "suboptimality_over_init_states_histories": np.zeros((num_runs, num_episodes)),
        "probability_assigned_to_optimal_actions_histories": np.zeros((num_runs, num_episodes, horizon, num_states)),
    } for lr in learning_rates}

    # Calculate the optimal value function of a given state or over initial distribution
    optimal_value, optimal_value_over_init_states = calculate_optimal_value_function(env)
    print(f"Optimal Value V*(s=0, h=0) = {optimal_value[0][0]}")
    print(f'Optimal Value V*(mu, h=0) = {optimal_value_over_init_states}')
    print("-" * 30)

    # Training loops
    for run in range(num_runs):
        print(f"\n==== Run {run + 1} ===")

        for lr in learning_rates:
            print(f"Training with LR={lr}")
            # --- Policy Parameters ---
            # A list where each element is the parameter table (logits) for that time step
            policy_params = [np.zeros((num_states, num_actions)) for _ in range(horizon)]
            suboptimality_history = []
            suboptimality_history_over_init_state = []
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

                value_function, value_function_over_init_state = calculate_value_function(env, policy_params)
                suboptimality = optimal_value[0][0] - value_function[0][0]
                suboptimality_over_init_state = optimal_value_over_init_states - value_function_over_init_state

                suboptimality_history.append(suboptimality)
                suboptimality_history_over_init_state.append(suboptimality_over_init_state)
                prob_action_1_snapshot = np.zeros((horizon, num_states))
                for h in range(horizon):
                    for s in range(num_states):
                        probs = softmax(policy_params[h][s, :])
                        prob_action_1_snapshot[h, s] = probs[1]
                prob_action_1_history.append(prob_action_1_snapshot)

            results_by_lr[lr]["suboptimality_histories"][run] = np.array(suboptimality_history)
            results_by_lr[lr]["suboptimality_over_init_states_histories"][run] = np.array(suboptimality_history_over_init_state)
            results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"][run] = np.array(
                prob_action_1_history)

            if is_save_ind_file:
                np.save(f'{model_path}subopt_lr_{lr}_run_{run}.npy', suboptimality_history)
                np.save(f'{model_path}subopt_over_init_state_lr_{lr}_run_{run}.npy',
                        suboptimality_history_over_init_state)
                np.save(f'{model_path}prob_to_opt_action_lr_{lr}_run_{run}.npy', np.array(prob_action_1_history))

    print("-" * 30)
    print("Training finished.")

    # Saving the training results
    if is_save_whole_file:
        np.save(f'{model_path}training_results.npy', results_by_lr)

    if is_save_plot and initial_learning_rates > 0:
        plot_average_subopt(num_runs, results_by_lr, learning_rates, "average_suboptimality", plot_path, episodes,
                            is_show_plot)
        plot_average_subopt(num_runs, results_by_lr, [1e-5, 1e-3, 0.1], "average_suboptimality_specific_lr", plot_path,
                            episodes, is_show_plot)
        plot_supopt_each_run(num_runs, results_by_lr, plot_path, episodes, is_show_plot)
        plot_opt_policy(results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
        plot_opt_policy_each_run(num_runs, results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)


if __name__ == "__main__":
    with open("./config/chain.yaml", "r") as f:
        config = yaml.safe_load(f)


    main(env=config['env_name'], chain_len=config['chain_len'], horizon=config['horizon'],
         learning_rates=list(config['learning_rates']),
         num_episodes=config['num_episodes'], num_runs=config['num_runs'], is_show_plot=config['is_show_plot'],
         is_save_plot=config['is_save_plot'], is_save_ind_file=config['is_save_ind_file'],
         is_save_whole_file=config['is_save_whole_file'])
