import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


@tf.keras.utils.register_keras_serializable()
class CriticNetwork(keras.Model):
    def __init__(self, action_dim, state_dim, fc1_dims=400, fc2_dims=300):
        super(CriticNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = keras.Sequential([keras.layers.Dense(units=self.fc1_dims),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Activation('relu')])

        self.fc2 = keras.layers.Dense(units=self.fc2_dims, activation='relu')

        self.q = keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3))

    def call(self, inputs, training=False):
        state, action = inputs
        x = self.fc1(state, training=training)
        x = tf.concat([x, action], axis=-1)
        x = self.fc2(x)
        q = self.q(x)
        return q

    def get_config(self):
        return {"action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "fc1_dims": self.fc1_dims,
                "fc2_dims": self.fc2_dims}


@tf.keras.utils.register_keras_serializable()
class ActorNetwork(keras.Model):
    def __init__(self, action_dim, state_dim, max_action, min_action, fc1_dims=400, fc2_dims=300):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action = max_action
        self.min_action = min_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        size_action = (self.max_action - self.min_action) / 2
        center_action = (self.max_action + self.min_action) / 2

        self.fc1 = keras.Sequential([keras.layers.Dense(units=self.fc1_dims),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Activation('relu')])

        self.fc2 = keras.Sequential([keras.layers.Dense(units=self.fc2_dims),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Activation('relu')])

        self.mu = keras.Sequential([keras.layers.Dense(units=self.action_dim,
                                                       kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3,
                                                                                                           maxval=3e-3),
                                                       activation=None),
                                    keras.layers.Activation('tanh'),
                                    keras.layers.Lambda(lambda z: z * size_action + center_action)])

    def call(self, state, training=False):
        x = self.fc1(state, training=training)
        x = self.fc2(x, training=training)
        mu = self.mu(x)
        return mu

    def get_config(self):
        return {"action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "max_action": self.max_action,
                "min_action": self.min_action,
                "fc1_dims": self.fc1_dims,
                "fc2_dims": self.fc2_dims}


@keras.utils.register_keras_serializable()
class CompensatorNetwork(keras.Model):
    def __init__(self, action_dim, state_dim, fc1_dims=30, fc2_dims=20):
        super(CompensatorNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = keras.layers.Dense(units=self.fc1_dims,
                                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                                      bias_initializer=keras.initializers.Zeros(), activation='relu')
        self.fc2 = keras.layers.Dense(units=self.fc2_dims,
                                      kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                                      bias_initializer=keras.initializers.Zeros(), activation='relu')
        self.q = keras.layers.Dense(units=self.action_dim,
                                    kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01),
                                    bias_initializer=keras.initializers.Zeros(), activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        u_bar = self.q(x)
        return u_bar

    def get_config(self):
        return {
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "fc1_dims": self.fc1_dims,
            "fc2_dims": self.fc2_dims}
