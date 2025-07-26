import numpy as np
import matplotlib.pyplot as plt

import numpy as np
def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


K = 3
r = np.array([2, 1, 1])
theta = np.zeros(K)
lr = 1

iter = 100
lambd = 0.2
ratio = []
norm = 2
pi_history = np.zeros((iter, K))
action_history = []
for t in range(iter):
  pi = softmax(theta)
  # delta_t = np.linalg.norm(r + lambd / K * 1 / pi, ord=norm)
  delta_t = np.linalg.norm(lambd * theta - r - np.sum(lambd * theta - r) / K * np.ones(K), ord=norm)
  # action_history.append(np.argmax(r + lambd / K * 1 / pi))
  # theta += lr * ((np.diag(pi) - pi @ pi.T) @ r + lambd * (1 / K - pi))
  theta += lr * ((np.diag(pi) - pi @ pi.T) @ (r - lambd * np.log(pi)))


  # Check if any norm delta_{t+1} \le delta_t
  pi = softmax(theta)
  # delta_t1 = np.linalg.norm(r + lambd / K * 1 / pi, ord=norm)
  delta_t1 =  np.linalg.norm(lambd * theta - r - np.sum(lambd * theta - r) / K * np.ones(K), ord=norm)
  # if delta_t1 != np.max(r + lambd / K * 1 /pi):
  #     break

  # if delta_t1 < delta_t:
  #     print("iteration: ", t)
      # print(f"{t}: {delta_t}          {t+1}: {delta_t1}")
  # print(f"iteration {t}: {delta_t}")
  ratio.append(np.log(delta_t1))
  pi_history[t] = pi
print('Final pi: ', softmax(theta))
print(action_history)

# plt.plot(np.arange(iter), pi_history[:, 0], label="$pi(1)$")
# plt.plot(np.arange(iter), pi_history[:, 1], label="$pi(2)$")
# plt.plot(np.arange(iter), pi_history[:, 2], label="$pi(3)$")
plt.plot(np.arange(iter), ratio, label=f"$||\\delta_t+1||/||\\delta_t||$")
plt.legend()
plt.title(f"$\lambda$ = {lambd}, lr ={lr}, iter = {iter}")
plt.savefig(f"./exp/entropy/lambda_{lambd}_norm_{norm}_lr_{lr}_iter_{iter}.png")
plt.show()
