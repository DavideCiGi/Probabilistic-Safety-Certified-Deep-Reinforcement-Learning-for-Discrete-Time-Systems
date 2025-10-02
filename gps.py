import gpflow
import itertools
import tensorflow as tf

from gpflow.kernels import RBF
from gpflow.kernels import Linear

from gpflow.utilities.model_utils import add_likelihood_noise_cov


def GP_Delta_kernel(state_dim, action_dim):
    sigma = gpflow.Parameter(1.0, trainable=False)

    kernel = (Linear(variance=sigma, active_dims=list(range(state_dim, state_dim + 1))) *
              RBF(lengthscales=[1.0] * state_dim, active_dims=list(range(state_dim))))

    for k in range(1, action_dim + 1):
        kernel += (Linear(variance=sigma, active_dims=list(range(state_dim + 1 + (k - 1),
                                                                 state_dim + 1 + k))) *
                   RBF(lengthscales=[1.0] * state_dim, active_dims=list(range(state_dim))))
    return kernel


def preliminary_computations_for_mnsrc(model):
    X_train = model.data[0]
    Y_train = model.data[1]
    state_dim = len(model.kernel.kernels[0].kernels[1].active_dims)

    beta_rows = tf.concat([tf.linalg.matrix_transpose(X_train[:, state_dim:state_dim + 1]),
                           tf.linalg.matrix_transpose(X_train[:, state_dim + 1:])], axis=0)

    ks = add_likelihood_noise_cov(model.kernel(X_train), model.likelihood, X_train)
    L_hat = tf.linalg.cholesky(ks)

    temp = tf.linalg.triangular_solve(L_hat, Y_train)
    m_right_factor = tf.linalg.triangular_solve(tf.linalg.matrix_transpose(L_hat), temp, lower=False)
    return beta_rows, m_right_factor, L_hat


def compute_mean_and_square_root_covariance(x, model, beta_rows, m_right_factor, L_hat, action_dim):
    X_train = model.data[0]

    x = tf.cast(model.state_normalizer(x), dtype=gpflow.config.default_float())
    lambda_row_list = []
    lambda_diag_list = []
    for j in range(1 + action_dim):
        lambda_row_list.append(model.kernel.kernels[j].kernels[1](x, X_train))
        lambda_diag_list.append(tf.squeeze(model.kernel.kernels[j].kernels[1](x, x))) 
    lambda_rows = tf.concat(lambda_row_list, axis=0)
    k_bar = lambda_rows * beta_rows
    lambda_diag_seq = tf.stack(lambda_diag_list, axis=0)
    lambda_diag = tf.linalg.diag(lambda_diag_seq)
    m = k_bar @ m_right_factor
    Sigma_factor_matrix = tf.linalg.triangular_solve(L_hat, tf.linalg.matrix_transpose(k_bar))
    Sigma = lambda_diag - tf.linalg.matrix_transpose(Sigma_factor_matrix) @ Sigma_factor_matrix

    Sigma = (Sigma + tf.transpose(Sigma)) / 2  # force the symmetry to avoid num. errors

    jitter = gpflow.config.default_jitter()
    jitter_eye = jitter * tf.eye(tf.shape(Sigma)[0], dtype=Sigma.dtype)
    Sigma_jittered = Sigma + jitter_eye

    L = tf.linalg.cholesky(Sigma_jittered)
    L_bar = tf.transpose(L)
    Lr_bar, L1_bar = tf.split(L_bar, [1, action_dim], axis=1)
    m_r, m_1 = tf.split(m, [1, action_dim], axis=0)

    return m_r, m_1, Lr_bar, L1_bar


