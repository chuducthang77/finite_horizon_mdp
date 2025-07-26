import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import argparse
import os
import yaml

from env.dice_mdp import DiceStopEnv
from utils.softmax import softmax
from utils.plot import *


def main(env, horizon, learning_rates, num_episodes, num_runs, plot_directory, is_show_plot=False):
    # Create the loader
    plot_path = "./exp/" + env + "/" + plot_directory + "/plots/"
    episodes = np.arange(1, num_episodes + 1)
    env_ = DiceStopEnv(horizon=horizon)
    num_states = env_.num_states
    results_by_lr = {lr: {
        "suboptimality_histories": np.zeros((num_runs, num_episodes)),
        "suboptimality_over_init_states_histories": np.zeros((num_runs, num_episodes)),
        "probability_assigned_to_optimal_actions_histories": np.zeros((num_runs, num_episodes, horizon, num_states)),
    } for lr in learning_rates}

    # Load the result
    titles = ["subopt_lr_", "subopt_over_init_state_lr_", "prob_to_opt_action_lr_"]
    for title in titles:
        for lr in learning_rates:
            for n in range(num_runs):
                # Load the results for each run
                if lr == 1e-5:
                    with open(f"./exp/{env}/{plot_directory}/models/{title}1e-05_run_{n}.npy", "rb") as f:
                        results = np.load(f)
                else:
                    with open(f"./exp/{env}/{plot_directory}/models/{title}{lr}_run_{n}.npy", "rb") as f:
                        results = np.load(f)
                # Store the results in the dictionary
                if title == "subopt_lr_":
                    results_by_lr[lr]["suboptimality_histories"][n] = results
                elif title == "subopt_over_init_state_lr_":
                    results_by_lr[lr]["suboptimality_over_init_states_histories"][n] = results
                elif title == "prob_to_opt_action_lr_":
                    results_by_lr[lr]["probability_assigned_to_optimal_actions_histories"][n] = results
            print('lr: ', lr)

    # Plot the result
    plot_average_subopt(num_runs, results_by_lr, learning_rates, "average_suboptimality", plot_path, episodes,
                        is_show_plot)
    plot_average_subopt(num_runs, results_by_lr, [1e-5, 1e-3, 0.1], "average_suboptimality_specific_lr", plot_path,
                        episodes, is_show_plot)
    plot_supopt_each_run(num_runs, results_by_lr, plot_path, episodes, is_show_plot)
    plot_opt_policy(results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)
    plot_opt_policy_each_run(num_runs, results_by_lr, 1000, learning_rates[:4], plot_path, is_show_plot)


if __name__ == "__main__":
    with open("./config/dice.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(env=config['env_name'], horizon=config['horizon'],
         learning_rates=list(config['learning_rates']), num_episodes=config['num_episodes'], num_runs=config['num_runs'],
                             plot_directory=config['plot_directory'], is_show_plot=config['is_show_plot'])
