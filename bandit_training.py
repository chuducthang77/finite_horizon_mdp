import numpy as np
import matplotlib.pyplot as plt
import pickle
# fixing the random seed
# np.random.seed(42)

K = 3 # number of arms
num_runs = 10 # number of runs to average across

# metrics to log and plot:
# pi is the prob of pulling the optimal action, grad is the gradient norm,
# action is a_t (arm sampled at round t), subopt is the suboptimality
# nastar is the sample count for the optimal action
metrics = ['pi']

T = 10**5 # number of iterations

eta_list = [1, 1e1, 1e2, 1e3] # list of step-sizes to run
# eta_list = [1]

# Arm 0 is the optimal arm
r = np.zeros(K)
r[0] = 0.2
r[1] = 0.2
r[2] = -0.1
# r[3] = -0.4

# dictionary to store the results
results = dict.fromkeys(eta_list)

def pi_map(K, theta):
    #theta = theta - np.max(theta, 1, keepdims=True)
    exp_theta = np.exp(theta)
    sum_exp_theta = np.sum(exp_theta)
    pi = exp_theta / sum_exp_theta
    return pi

def one_run(T, eta, theta_0):

  theta = theta_0 # initialize

  pi_1_list = []
  pi_2_list = []
  pi_optimal_total_list = []

  for t in range(T):

    # GET POLICY
    pit = pi_map(K, theta)

    # SAMPLE ACTION
    a_t = int(np.random.choice(np.arange(K), 1 , p=pit) [0])


    # GET REWARD
    r_t = r[a_t]
    r_t = r[a_t] + 0.1 * np.random.randn()

    # UPDATE
    theta[a_t] = theta[a_t] + eta * r_t
    theta = theta - eta * pit * r_t

    # LOGGING
    pi_1_list.append(pit[0])
    pi_2_list.append(pit[1])
    pi_optimal_total_list.append(pit[0] + pit[1])

  return pi_1_list, pi_2_list, pi_optimal_total_list

for eta in eta_list:

  results[eta] = dict.fromkeys(metrics)

  # for logging
  pi_list_runs_a1 = np.zeros((num_runs, T))
  pi_list_runs_a2 = np.zeros((num_runs, T))
  pi_list_runs_total_optimal = np.zeros((num_runs, T))

  for run in range(num_runs):

    print('eta = ', eta, 'Run = ', run)
    p1,p2,pi_total = one_run(T, eta, np.zeros(K))
    #p,g,a,s = one_run(T, eta, [np.log(0.08), np.log(0.92)])


    pi_list_runs_a1[run, :] = p1
    pi_list_runs_a2[run, :] = p2
    pi_list_runs_total_optimal[run, :] = pi_total

  results[eta]['pi1'] = pi_list_runs_a1
  results[eta]['pi2'] = pi_list_runs_a2
  results[eta]['pi_optimal_total'] = pi_list_runs_total_optimal



# plot the mean across runs
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes = axes.flatten()
for idx, eta in enumerate(eta_list):
  ax = axes[idx]
  ax.plot(np.mean(results[eta]['pi1'], axis = 0), label=r"$\pi_{\theta_t}(a_1)$")
  ax.plot(np.mean(results[eta]['pi2'], axis = 0), label=r"$\pi_{\theta_t}(a_2)$")
  ax.plot(np.mean(results[eta]['pi_optimal_total'], axis=0), label=r"$\pi_{\theta_t}(a_1) + \pi_{\theta_t}(a_2)$")
  # plt.yscale('log')
  # ax.set_ylim(0.3, 1.05)
  ax.set_ylabel(r'Probability of optimal arms $\pi_{\theta_t}(a^*)$')
  ax.set_xlabel('Episodes (t)')
  ax.grid(True)
  ax.set_title(f"$\\eta = {eta}$")
  #plt.legend('Probability of optimal action')
  ax.legend()

plt.suptitle(r"Total probability of optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*)$")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./bandit/large_learning_rate_total_prob_astar_eta_{eta:.2f}_over_{run}_runs.pdf".format(eta=eta, run=num_runs), dpi=3000)


for n in range(num_runs):
  plt.figure(figsize=(10, 6))
  for eta in eta_list:
    plt.plot(results[eta]['pi1'][n, :], label=r"$\pi_{\theta_t}(a_1)$")
    plt.plot(results[eta]['pi2'][n, :], label=r"$\pi_{\theta_t}(a_2)$")
    plt.plot(results[eta]['pi_optimal_total'][n, :], label=r"$\pi_{\theta_t}(a_1) + \pi_{\theta_t}(a_2)$")
    plt.ylabel('Probability of optimal arm')
    plt.xlabel('Episodes (t)')
    plt.grid(True)
    plt.title(r"Total probability of optimal actions $\sum_{a^*}\pi_{\theta_t}(a^*)$")
    plt.legend()
    plt.savefig("./bandit/large_learning_rate_total_prob_astar_eta_{eta:.2f}_run_{n}.pdf".format(eta=eta, n=n), dpi=3000)
    plt.close()

with open('./bandit/data.pkl', 'wb') as f:
    pickle.dump(results, f)