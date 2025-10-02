import numpy as np


class DReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):  # input shape of observation from our environment
        # action_dim --> It's the number of components of the action.
        self.mem_size = max_size
        self.mem_cntr = 0  # it's a memory counter, initiated to zero
        self.state_memory = np.zeros((self.mem_size, state_dim))  # the star * just unpack a list of elements
        self.new_state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)  # we're going to use it as a mask in the agent

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1
        if self.mem_cntr == self.mem_size:
            print('Replay buffer for the agent filled up for the first time.')

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones


class AReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.compensation_memory = np.zeros((self.mem_size, action_dim))

    def store_transition(self, state, compensation):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.compensation_memory[index] = compensation

        self.mem_cntr += 1
        if self.mem_cntr == self.mem_size:
            print('State-action replay for the compensator filled up for the first time.')

    def get_all(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        indices = np.arange(max_mem)

        states = self.state_memory[indices]
        compensations = self.compensation_memory[indices]

        return states, compensations


class GPBuffer:
    def __init__(self, max_size, state_dim, action_dim, episode_length):
        if max_size < episode_length:  # technically we should subtract r and the number of times the patient has
            # eaten in episode_length steps (if it doesn't coincide with the first r steps) since we're sampling the
            # buffer at the end of each episode (otherwise what to subtract becomes more convoluted)
            self.mem_size = episode_length  # we'll memorize more step but we'll draw randomly max_size still
        else:
            self.mem_size = max_size
        self.max_size = max_size
        self.mem_cntr = 0
        self.episode_length = episode_length
        self.X_memory = np.zeros((self.mem_size, state_dim + action_dim))
        self.Y_psi_memory = np.zeros((self.mem_size, 1))
        self.Y_phi_memory = np.zeros((self.mem_size, 1))

    def store(self, state_action, psi_error, phi_error):
        index = self.mem_cntr % self.mem_size
        self.X_memory[index] = state_action
        self.Y_psi_memory[index] = psi_error
        self.Y_phi_memory[index] = phi_error

        self.mem_cntr += 1
        if self.mem_cntr == self.mem_size:
            print('Gaussian Process training batch filled up for the first time.')

    def special_sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_size = min(self.mem_cntr, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        X = self.X_memory[batch]
        Y_psi = self.Y_psi_memory[batch]
        Y_phi = self.Y_phi_memory[batch]

        return X, Y_psi, Y_phi
