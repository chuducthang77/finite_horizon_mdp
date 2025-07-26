class TreeEnv:
    def __init__(self, branching_factor=3, depth=3):
        self.reset()

        # Define the deterministic transition structure as a dict:
        # (current_state, action) -> (next_state, reward)
        self.num_states = 0
        self.num_terminal_states = 0
        self.depth = depth
        self.num_actions = branching_factor
        self.terminal_states = None
        self.transitions = {}
        self.build_transition()


    def reset(self):
        self.current_state = 0
        self.done = False
        return self.current_state

    def reward_set(self, state, action, next_state, reward):
        self.transitions[(state, action)] = (next_state, reward)

    def build_transition(self):
        self.transitions = {}
        current_state = 0
        current_depth = 0
        next_state = 1
        queue = [(current_state, current_depth)]
        while len(queue) > 0:
            curr_state, curr_depth = queue.pop(0)
            if curr_depth < self.depth:
                for action in range(self.num_actions):
                    self.transitions[(curr_state, action)] = (next_state, 0)
                    next_state += 1
                    queue.append((curr_state, curr_depth + 1))

        for i in range(depth):
            self.num_states += branching_factor ** i

        self.num_terminal_states = branching_factor ** (depth - 1)

        self.terminal_states = set(range(self.num_states - self.num_terminal_states, self.num_states))  # Leaf nodes


    def step(self, action):
        if self.done:
            raise Exception("Episode already finished.")
        key = (self.current_state, action)
        if key not in self.transitions:
            raise ValueError(f"Invalid action {action} at state {self.current_state}")

        next_state, reward = self.transitions[key]
        self.current_state = next_state
        self.done = next_state in self.terminal_states
        return next_state, reward, self.done
