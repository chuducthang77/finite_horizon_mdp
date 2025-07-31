import numpy as np
from datetime import datetime
import os
import yaml
import argparse

from environment.deep_sea_mdp import DeepSeaEnv
from utils.softmax import softmax
from utils.plot import *


def round_significant(number, significant_figures=6):
    """
    Rounds a float to a specific number of significant figures.
    """
    if number == 0:
        return 0

    # Use string formatting to round to the specified number of significant digits
    # and then convert the resulting string back to a float.
    format_string = f"{{:.{significant_figures}g}}"
    return float(format_string.format(number))


def main(env, grid_size, horizon, learning_rates, num_episodes, num_runs, dir_title, plot_directory, is_show_plot=False,
         is_save_subopt=False, is_save_subopt_over_init_state=True, is_save_prob_optimal_action=False,
         is_save_subopt_over_init_state_last_iterate=False):
    # Create the loader
    plot_path = "./exp/" + env + "/" + dir_title + "/" + plot_directory + "/plots/"
    episodes = np.arange(1, num_episodes + 1)
    env_ = DeepSeaEnv(grid_size=grid_size, horizon=horizon)
    num_states = env_.num_states
    initial_learning_rates = len(learning_rates)

    # learning_rates = np.delete(learning_rates, 2)
    temp_learning_rates = []
    for lr in learning_rates:
        temp_learning_rates.append(round_significant(lr))
    learning_rates = np.array(temp_learning_rates)

    results_by_lr = {lr: {
        "suboptimality_over_init_states_histories": np.zeros((num_runs, num_episodes)),
    } for lr in learning_rates}

    if is_save_subopt:
        for lr in learning_rates:
            results_by_lr[lr]['suboptimality_histories'] = np.zeros((num_runs, num_episodes))

    if is_save_prob_optimal_action:
        for lr in learning_rates:
            results_by_lr[lr]['probability_assigned_to_optimal_actions_histories'] = np.zeros((num_runs, num_episodes, horizon, num_states))

    # Load the result
    if is_save_subopt and is_save_prob_optimal_action:
        titles = ["subopt_lr_", "subopt_over_init_state_lr_", "prob_to_opt_action_lr_"]
    elif is_save_subopt and not is_save_prob_optimal_action:
        titles = ["subopt_lr_", "subopt_over_init_state_lr_"]
    elif not is_save_subopt and is_save_prob_optimal_action:
        titles = ["subopt_over_init_state_lr_", "prob_to_opt_action_lr_"]
    elif not is_save_subopt and not is_save_prob_optimal_action:
        titles = ["subopt_over_init_state_lr_"]
    elif is_save_subopt_over_init_state_last_iterate:
        titles = ['subopt_over_init_state_lr_']

    for title in titles:
        for lr in learning_rates:
            for n in range(num_runs):
                # Load the results for each run
                if lr == 1e-5:
                    with open(f"./exp/{env}/{dir_title}/{plot_directory}/models/{title}1e-05_run_{n}.npy", "rb") as f:
                        results = np.load(f)
                elif lr == 1e-8:
                    with open(f"./exp/{env}/{dir_title}/{plot_directory}/models/{title}1e-08_run_{n}.npy", "rb") as f:
                        results = np.load(f)
                else:
                    with open(f"./exp/{env}/{dir_title}/{plot_directory}/models/{title}{lr}_run_{n}.npy", "rb") as f:
                        results = np.load(f)
                # Store the results in the dictionary
                if title == "subopt_lr_":
                    results_by_lr[lr]["suboptimality_histories"][n] = results
                if title == "subopt_over_init_state_lr_":
                    results_by_lr[lr]["suboptimality_over_init_states_histories"][n] = results
                elif title == "prob_to_opt_action_lr_":
                    results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"][n] = results
            print('lr: ', lr)

    # Plot the result
    if is_save_subopt:
        plot_supopt_each_run(num_runs, results_by_lr, plot_path, episodes, is_show_plot)
    if is_save_prob_optimal_action:
        plot_opt_policy(results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
        plot_opt_policy_each_run(num_runs, results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
    if not is_save_subopt_over_init_state_last_iterate:
        plot_average_subopt(num_runs, results_by_lr, learning_rates, "average_suboptimality", plot_path, episodes,
                            is_show_plot)
        # plot_average_subopt(num_runs, results_by_lr, [1e-5, 1e-3, 0.1], "average_suboptimality_specific_lr", plot_path,
        #                     episodes, is_show_plot)
    else:
        plot_average_supopt_last_iterate(num_runs, results_by_lr, learning_rates, episodes,
                                         "average_suboptimality_last_iterate", plot_path, is_show_plot)


if __name__ == "__main__":
    with open("./config/deep_sea.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='default')
    parsed_args = parser.parse_args()

    if len(list(config['learning_rates'])) > 0:
        learning_rates = list(config['learning_rates'])
    else:
        learning_rates = np.exp(np.linspace(-9, 0, num=50))

    main(env=config['env_name'], grid_size=config['grid_size'], horizon=config['horizon'],
         learning_rates=learning_rates, num_episodes=config['num_episodes'], num_runs=config['num_runs'],
         dir_title=parsed_args.title, plot_directory=config['plot_directory'], is_show_plot=config['is_show_plot'],
         is_save_subopt=config['is_save_subopt_histories'],
         is_save_subopt_over_init_state=config['is_save_subopt_over_init_state_histories'],
         is_save_prob_optimal_action=config['is_save_prob_optimal_action_histories'],
         is_save_subopt_over_init_state_last_iterate=config[
             'is_save_subopt_over_init_state_last_iterate'])
