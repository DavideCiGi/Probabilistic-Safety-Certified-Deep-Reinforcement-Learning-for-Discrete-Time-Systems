import tensorflow as tf
from tensorflow import keras
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer


def polyak_update(params, target_params, tau):
    for param, target_param in zip(params, target_params):
        target_param.assign(tau * param + (1.0 - tau) * target_param)


class Agent:
    def __init__(self, state_dim, action_dim, max_action, min_action, alr=1e-4, clr=1e-3,
                 gamma=0.99, max_size=1_000_000, tau=5e-3, d=2, explore_sigma=0.1, smooth_sigma=0.2, c=0.5,
                 fc1_dims=400, fc2_dims=300, batch_size=128, chkpt_dir='models/td3/'):
        self.c = c
        self.smooth_sigma = smooth_sigma
        self.explore_sigma = explore_sigma
        self.gamma = gamma
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.d = d
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        self.learn_step_cntr = 0

        self.actor = ActorNetwork(action_dim=action_dim, state_dim=state_dim, max_action=self.max_action,
                                  min_action=self.min_action, fc1_dims=fc1_dims,
                                  fc2_dims=fc2_dims)
        self.critic_1 = CriticNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic_2 = CriticNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_actor = ActorNetwork(action_dim=action_dim, state_dim=state_dim, max_action=self.max_action,
                                         min_action=self.min_action, fc1_dims=fc1_dims,
                                         fc2_dims=fc2_dims)
        self.target_critic_1 = CriticNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims,
                                             fc2_dims=fc2_dims)
        self.target_critic_2 = CriticNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims,
                                             fc2_dims=fc2_dims)

        # Consider gradient clipping because GPs have limited memory, and we can't afford to let the actors grow
        # too much by exploring state-actions that the GPs haven't seen or no longer have in memory!
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=alr, clipnorm=5.)
        self.critic_1_optimizer = keras.optimizers.Adam(learning_rate=clr, clipnorm=5.)
        self.critic_2_optimizer = keras.optimizers.Adam(learning_rate=clr, clipnorm=5.)

        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, action_dim), dtype=tf.float32)

        self.actor(dummy_state)
        self.critic_1((dummy_state, dummy_action))
        self.critic_2((dummy_state, dummy_action))
        self.target_actor(dummy_state)
        self.target_critic_1((dummy_state, dummy_action))
        self.target_critic_2((dummy_state, dummy_action))

        optimizers_to_prime = [(self.actor_optimizer, self.actor.trainable_variables),
                               (self.critic_1_optimizer, self.critic_1.trainable_variables),
                               (self.critic_2_optimizer, self.critic_2.trainable_variables)]

        for optimizer, params in optimizers_to_prime:
            if params:
                dummy_grads = [tf.zeros_like(p) for p in params]
                optimizer.apply_gradients(zip(dummy_grads, params))

        self.update_network_trainable_parameters(tau=1)
        self.update_network_non_trainable_parameters()

        self.ckpt = tf.train.Checkpoint(actor=self.actor,
                                        critic_1=self.critic_1,
                                        critic_2=self.critic_2,
                                        target_actor=self.target_actor,
                                        target_critic_1=self.target_critic_1,
                                        target_critic_2=self.target_critic_2,
                                        actor_optimizer=self.actor_optimizer,
                                        critic_1_optimizer=self.critic_1_optimizer,
                                        critic_2_optimizer=self.critic_2_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.chkpt_dir, max_to_keep=3)

    def save_models(self):
        if self.memory.mem_cntr < self.batch_size:
            # agent doesn't go into learning, no need to save the parameters (and it's not possible regardless until
            # the agent goes into learning which happens when buffer has at least one possible batch).
            return False
        else:
            print('... saving models ...')
            self.ckpt_manager.save()
            return True

    def load_models(self):
        print('... loading models ...')
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print(f'Model restored from latest checkpoint.')
        else:
            print('No checkpoint found.')

    def collect_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, observation, evaluate=False, explore_sigma=None):
        if explore_sigma is None:
            explore_sigma = self.explore_sigma
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu = self.actor(state)[0]
        if not evaluate:  # it might be good to ensure the exploration doesn't give zero at all for stability
            mu += tf.random.normal(shape=[self.action_dim], mean=0.0, stddev=explore_sigma)
        mu = tf.clip_by_value(mu, self.min_action, self.max_action)
        return mu

    def update_network_trainable_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        polyak_update(self.actor.trainable_variables, self.target_actor.trainable_variables, tau)
        polyak_update(self.critic_1.trainable_variables, self.target_critic_1.trainable_variables, tau)
        polyak_update(self.critic_2.trainable_variables, self.target_critic_2.trainable_variables, tau)

    def update_network_non_trainable_parameters(self, tau=None):
        if tau is None:
            tau = 1.

        polyak_update(self.actor.non_trainable_variables, self.target_actor.non_trainable_variables, tau)
        polyak_update(self.critic_1.non_trainable_variables, self.target_critic_1.non_trainable_variables, tau)
        polyak_update(self.critic_2.non_trainable_variables, self.target_critic_2.non_trainable_variables, tau)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        self.update_critics(states, actions, rewards, next_states, dones)
        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.d == 0:
            self.update_actor(states)
            self.update_network_trainable_parameters()
            self.update_network_non_trainable_parameters()

    @tf.function
    def update_critics(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            target_actions = self.target_actor(next_states)

            epsilons = tf.clip_by_value(tf.random.normal(
                shape=tf.shape(target_actions), stddev=self.smooth_sigma), -self.c, self.c)

            smooth_actions = target_actions + epsilons
            smooth_actions = tf.clip_by_value(smooth_actions, self.min_action,
                                              self.max_action)

            next_critic_1_value = tf.squeeze(self.target_critic_1((next_states, smooth_actions)), axis=-1)
            next_critic_2_value = tf.squeeze(self.target_critic_2((next_states, smooth_actions)), axis=-1)
            target = rewards + self.gamma * tf.math.minimum(next_critic_1_value, next_critic_2_value) * (1 - dones)
            target = tf.stop_gradient(target)
            critic_1_value = tf.squeeze(self.critic_1((states, actions), training=True), axis=-1)
            critic_2_value = tf.squeeze(self.critic_2((states, actions), training=True), axis=-1)
            critic_1_loss = tf.reduce_mean(tf.square(target - critic_1_value))
            critic_2_loss = tf.reduce_mean(tf.square(target - critic_2_value))
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape1.gradient(critic_1_loss, params_1)
        grads_2 = tape2.gradient(critic_2_loss, params_2)

        self.critic_1_optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2_optimizer.apply_gradients(zip(grads_2, params_2))

    @tf.function
    def update_actor(self, states):
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            new_critic_1_value = tf.squeeze(self.critic_1((states, new_actions)), axis=-1)
            actor_loss = tf.math.reduce_mean(- new_critic_1_value)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor_optimizer.apply_gradients(zip(grads, params))
