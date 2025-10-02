import tensorflow as tf
from tensorflow import keras
from networks import CompensatorNetwork
from buffer import AReplayBuffer
import numpy as np


class BarrierCompensator:
    def __init__(self, state_dim, action_dim, max_action, min_action, bar_constraint_max=0.02, max_size=15_000,
                 fc1_dims=30, fc2_dims=20, epochs=10, num_backtracking=10, chkpt_dir='models/comp/'):
        self.action_dim = action_dim
        self.max_abs_action = max(max_action, min_action)
        self.memory = AReplayBuffer(max_size, state_dim, action_dim)
        self.chkpt_dir = chkpt_dir
        self.epochs = epochs
        self.bar_constraint_max = bar_constraint_max
        self.num_backtracking = num_backtracking

        self.loss_fn = keras.losses.MeanSquaredError()

        self.compensator = CompensatorNetwork(action_dim=action_dim, state_dim=state_dim, fc1_dims=fc1_dims,
                                              fc2_dims=fc2_dims)

        self.ckpt = tf.train.Checkpoint(compensator=self.compensator)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.chkpt_dir, max_to_keep=3)

    def save_model(self):
        print('... saving models ...')
        self.ckpt_manager.save()
        return True

    def load_model(self):
        print('... loading models ...')
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print(f'Model restored from latest checkpoint.')
        else:
            print('No checkpoint found.')

    def collect_transition(self, state, action):
        self.memory.store_transition(state, action)

    def compensation(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        u_bar = self.compensator(state)[0]
        u_bar = tf.clip_by_value(u_bar, -self.max_abs_action, self.max_abs_action)
        return u_bar

    def learn(self):
        states, compensations = self.memory.get_all()

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        compensations = tf.convert_to_tensor(compensations, dtype=tf.float32)

        self.update_compensator(states, compensations)

    def update_compensator(self, states, compensations):

        success_any = False
        pred = self.compensator(states, training=False)
        current_loss = self.loss_fn(compensations, pred)
        initial_loss_val = float(current_loss.numpy())

        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                pred_compensations = self.compensator(states, training=True)
                current_loss = self.loss_fn(compensations, pred_compensations)

            tr_variables = self.compensator.trainable_variables
            gradients = tape.gradient(current_loss, tr_variables)

            safe_gradients = []
            for var, g in zip(tr_variables, gradients):
                if g is None:
                    safe_gradients.append(tf.zeros_like(var))
                else:
                    if g.dtype != var.dtype:
                        safe_gradients.append(tf.cast(g, var.dtype))
                    else:
                        safe_gradients.append(g)

            full_step = []
            for g in safe_gradients:
                step_scale = tf.convert_to_tensor(self.bar_constraint_max, dtype=g.dtype)
                full_step.append(-step_scale * g)

            epoch_accepted = False

            old_vals = [tf.identity(v) for v in tr_variables]

            for fraction in 0.5 ** np.arange(self.num_backtracking):
                step_frac = [tf.cast(fraction, step.dtype) * step for step in full_step]

                # temporary update
                for tr_var, update in zip(tr_variables, step_frac):
                    tr_var.assign_add(update)

                new_pred_compensations = self.compensator(states, training=True)
                new_loss_val = float(self.loss_fn(compensations, new_pred_compensations).numpy())

                if new_loss_val < initial_loss_val:
                    # accept update for this epoch
                    initial_loss_val = new_loss_val
                    epoch_accepted = True
                    success_any = True
                    break  # go to the next epoch (don't restore old values)
                else:
                    # efficient rollback (old values restored) and next fraction trial
                    for v, old in zip(tr_variables, old_vals):
                        v.assign(old)

            if not epoch_accepted:
                # no fraction has improved the loss: keep the old weights
                for v, old in zip(tr_variables, old_vals):
                    v.assign(old)
                # initial_loss_val remain unaltered

        return success_any, initial_loss_val
