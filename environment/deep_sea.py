import numpy as np

# Chain MDP environment
class ChainEnv:
    def __init__(self, chain_length=5, horizon=10, exit_reward=0.5, intermediate_reward = -0.5, final_reward = 7):
        self.chain_length = chain_length
        self.horizon = horizon
        self.num_states = chain_length + 1  # Last state is terminal
        self.num_actions = 2  # 0: terminate, 1: continue
        self.terminal_state = self.chain_length
        self.exit_reward = exit_reward
        self.intermediate_reward = intermediate_reward
        self.final_reward = final_reward
        self.reset()
        self.reward_check()

    def reward_check(self):
        if self.final_reward + self.chain_length * self.intermediate_reward <= self.exit_reward:
            raise Exception("The immediate reward is more than the final reward.")

    def reset(self):
        self.current_state = np.random.randint(0, self.chain_length)
        self.current_step = 0
        self.done = False
        return self.current_state

    def step(self, action):
        if self.done or self.current_step >= self.horizon:
            raise Exception("Episode already finished.")

        if action not in [0, 1]:
            raise ValueError("Invalid action")

        if action == 0:
            # Terminate immediately
            next_state = self.terminal_state
            reward = self.exit_reward
        else:  # action == 1
            next_state = self.current_state + 1
            if next_state >= self.chain_length:
                next_state = self.terminal_state

            # Reward 1 only if we reach the terminal state from the last state
            reward = self.final_reward if (self.current_state == self.chain_length - 1 and action == 1) else self.intermediate_reward

        self.current_state = next_state
        self.current_step += 1
        self.done = self.current_state == self.terminal_state

        return next_state, reward, self.done