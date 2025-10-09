import numpy as np
from agent import Agent
from compensator import BarrierCompensator
from gps import (GP_Delta_kernel, compute_mean_and_square_root_covariance,
                 preliminary_computations_for_mnsrc)
from utils import (plot_learning_curve, manage_memory, DT_Bergman_dynamics, plot_evaluation_run_with_GPs,
                   reward_calc, plot_reward_function, plot_violation_curve, meal_schedule, ExplorationPolicy)
import gpflow
import tensorflow as tf
import time
import shutil
import os

answer0 = input("Do you want to use CVXPYgen to accelerate the SOCP? (y/n) ")
while not (answer0 == 'yes' or answer0 == 'Yes' or answer0 == 'YES' or answer0 == 'y' or answer0 == 'Y'
           or answer0 == 'no' or answer0 == 'No' or answer0 == 'NO' or answer0 == 'n' or answer0 == 'N'):
    answer0 = input("Please provide a yes or a no as an answer! ")
if answer0 == 'yes' or answer0 == 'Yes' or answer0 == 'YES' or answer0 == 'y' or answer0 == 'Y':
    from cbfcontroller_gen import (DT_CBFs, CBF_with_GPs_SOCP)
else:
    from cbfcontroller import (DT_CBFs, CBF_with_GPs_SOCP)

if __name__ == '__main__':
    # Our knowledge about dynamics: nominal model dynamics
    nominal_model_params = {
        'p_1': 2.3e-6,
        'G_b': 75.0,
        'p_2': 0.088,
        'p_3': 0.63e-3,
        'n': 0.09,
        'I_b': 15.0,
        'tau_G': 47.0,
        'V_G': 253.0
    }

    # CBF1_params smaller values than CBF2_params because not only I need to avoid hypoglycemia at all costs,
    # but, most importantly, because action_max is well enough to deal with any BG spike, while action_min cannot deal
    # with any trough the system will face, it would hope to provide negative values, but the constraint force it not.
    CBF1_params = {
        'gamma_11': 0.175,
        'gamma_21': 0.150,
        'gamma_31': 0.125
    }

    CBF2_params = {
        'gamma_12': 0.325,
        'gamma_22': 0.300,
        'gamma_32': 0.275
    }

    np.random.seed(42)

    # the true model dynamics
    true_model_params = {
        k: v if k in ('tau_G', 'V_G')
        else np.random.choice([0.7, 1.3]) * v
        for k, v in nominal_model_params.items()
    }
    print(f'\nThe true model parameters:\n{true_model_params}\n')

    # every time should be thought in minutes, unless specified differently
    dt = 1  # 0 < dt < 60

    N = 24 * 60
    print(f'Total number of steps: {N}.')
    CHO_max = 65_000  # YOU ASSUME TO KNOW THE MAXIMAL CHO INTAKE

    CHOs, eating = meal_schedule(N, dt, CHO_max)

    np.random.seed(None)

    answer1 = input("Do you want to do an evaluation run? (y/n) ")
    while not (answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES' or answer1 == 'y' or answer1 == 'Y'
               or answer1 == 'no' or answer1 == 'No' or answer1 == 'NO' or answer1 == 'n' or answer1 == 'N'):
        answer1 = input("Please provide a yes or a no as an answer! ")
    if answer1 == 'yes' or answer1 == 'Yes' or answer1 == 'YES' or answer1 == 'y' or answer1 == 'Y':
        evaluate = True
        restore_training = False
    else:
        evaluate = False
        answer2 = input("Do you want to do restore training? (y/n) ")
        while not (answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES' or answer2 == 'y' or answer2 == 'Y'
                   or answer2 == 'no' or answer2 == 'No' or answer2 == 'NO' or answer2 == 'n' or answer2 == 'N'):
            answer2 = input("Please provide a yes or a no as an answer! ")
        if answer2 == 'yes' or answer2 == 'Yes' or answer2 == 'YES' or answer2 == 'y' or answer2 == 'Y':
            restore_training = True
        else:
            restore_training = False

    action_dim = 1
    state_dim = 5
    max_action = 30.  # important to guarantee it's a float
    min_action = 0.

    max_action_agent = 0.3
    min_action_agent = min_action
    size_action_agent = (max_action_agent - min_action_agent) / 2
    center_action_agent = (max_action_agent + min_action_agent) / 2
    compensator_tr_games = 10

    manage_memory()
    agent = Agent(state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action_agent,
                  min_action=min_action_agent,
                  alr=1e-4, clr=1e-3,
                  max_size=1_000_000, tau=5e-3, d=2, explore_sigma=0.1 * size_action_agent,
                  smooth_sigma=0.2 * size_action_agent, c=0.5 * size_action_agent, fc1_dims=400, fc2_dims=300,
                  batch_size=128)
    os.makedirs(agent.chkpt_dir, exist_ok=True)

    compensator = BarrierCompensator(state_dim=state_dim,
                                     action_dim=action_dim,
                                     max_action=max_action,
                                     min_action=min_action, bar_constraint_max=0.02,
                                     max_size=N * (compensator_tr_games + 1), fc1_dims=30, fc2_dims=20,
                                     epochs=10, num_backtracking=10)
    os.makedirs(compensator.chkpt_dir, exist_ok=True)

    GP_max_size = 1000
    state_normalizer = tf.keras.layers.Normalization()
    dt_cbfs = DT_CBFs(dt, nominal_model_params, CBF1_params, CBF2_params, CHO_max, GP_max_size, N, true_model_params)

    kernels_psi = GP_Delta_kernel(state_dim, action_dim)
    kernels_phi = GP_Delta_kernel(state_dim, action_dim)
    GP_psi_dir = 'models/GP_psi'
    GP_phi_dir = 'models/GP_phi'

    k_delta = 1.5
    K_eps_1 = 1e6
    K_eps_2 = 1e6
    cbf_socp = CBF_with_GPs_SOCP(k_delta, K_eps_1, K_eps_2, max_action, min_action, CBF1_params, CBF2_params)
    # a change in the parameters above require necessarily the SOCP problem to be re-written again in C

    G0 = 140
    best_score = reward_calc(50) * N  # value that will get updated ofc
    print(f'Worst case scenario score: {best_score}.')
    print(f'Best policy reward: {reward_calc(G0) * N}.')  # best reward we can hope
    reward_avg_window = 50
    score_history = []
    max_violation_history = []

    if evaluate:
        n_games = 1

        states = []
        controls = []
        worst_CBF_psi = []
        worst_CBF_phi = []
        mean_CBF_psi = []
        mean_CBF_phi = []
        nominal_CBF_psi = []
        nominal_CBF_phi = []
        epsilon_psi = []
        epsilon_phi = []
        true_CBF_psi = []
        true_CBF_phi = []
        u_RL_controls = []
        u_bar_controls = []
        u_CBF_controls = []

        agent.load_models()
        compensator.load_model()  # ATTENTION when restoring training: assure the previous training had time to at
        # least pass the 5th episode!

    else:
        n_games = 200
        print(f'Number of episodes: {n_games}.')
        os.makedirs("plots", exist_ok=True)
        plot_reward_function(figure_file='plots/RewardFunction.png')
        if not restore_training:
            shutil.rmtree(compensator.chkpt_dir)
            os.makedirs(compensator.chkpt_dir)
        else:
            agent.load_models()
            compensator.load_model()

        final_sigma = 1e-4
        exploration_policy = ExplorationPolicy(0.1 * size_action_agent, n_games, final_sigma, 1e-6, 20)

    time.sleep(5)

    for j in range(n_games):
        if j == compensator_tr_games + 1:
            print('\nFrom now on, you can restore correctly the compensator!\n')
            time.sleep(3)
        print(f'Episode {j} started.')

        if evaluate or (j == 0 and restore_training):
            psi_model = tf.saved_model.load(GP_psi_dir + '/restored')
            phi_model = tf.saved_model.load(GP_phi_dir + '/restored')
        elif j > 0:
            psi_model = tf.saved_model.load(GP_psi_dir)
            phi_model = tf.saved_model.load(GP_phi_dir)

        score = 0
        episode_violations = []
        state = np.array([G0, 0., true_model_params['I_b'], 0., 0.])
        if evaluate:
            explore_sigma = None
        else:
            explore_sigma = exploration_policy.get_sigma()

        for i in range(N):
            episode_violations.append(max(0., -state[0] + dt_cbfs.BG_min_value, +state[0] - dt_cbfs.BG_max_value))
            print(f'Step {i} started (episode {j}).')
            # print(f'Current state: {state}')
            print(f'Current blood glucose level: {state[0]:.5f}')
            u_RL = agent.choose_action(state, evaluate, explore_sigma)[0].numpy()
            print(f'u_RL: {u_RL:.5f}')

            if j > 0 or evaluate or restore_training:
                u_bar = compensator.compensation(state)[0].numpy()
                print(f'u_bar: {u_bar:.5f}')

                (psi_m_r_temp, psi_m_1_temp, psi_Lr_bar_temp,
                 psi_L1_bar_temp) = psi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(psi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(psi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_psi!")

                (phi_m_r_temp, phi_m_1_temp, phi_Lr_bar_temp,
                 phi_L1_bar_temp) = phi_model.compiled_mean_and_square_root_covariance(state.reshape((1, 5)))

                if tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(phi_Lr_bar_temp)),
                                           tf.reduce_any(tf.math.is_nan(phi_L1_bar_temp))])):
                    tf.print("\n!!! Cholesky produced NaNs for Sigma_phi!")

                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (psi_m_r_temp.numpy(), psi_m_1_temp.numpy(),
                                                            psi_Lr_bar_temp.numpy(), psi_L1_bar_temp.numpy())

                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (phi_m_r_temp.numpy(), phi_m_1_temp.numpy(),
                                                            phi_Lr_bar_temp.numpy(), phi_L1_bar_temp.numpy())

            else:  # basically if we are in the first step of the training phase and not in evaluation,
                # we restrict ourselves to a simple QP
                u_bar = 0.
                psi_m_r, psi_m_1, psi_Lr_bar, psi_L1_bar = (np.zeros((1, 1)),  # ----> CORRECT IT, r -> 1, the rest?
                                                            np.zeros((1, action_dim)), np.zeros((1 + action_dim, 1)),
                                                            np.zeros((1 + action_dim, action_dim)))
                phi_m_r, phi_m_1, phi_Lr_bar, phi_L1_bar = (np.zeros((1, 1)),
                                                            np.zeros((1, action_dim)), np.zeros((1 + action_dim, 1)),
                                                            np.zeros((1 + action_dim, action_dim)))

            a_11, a_21, a_12, a_22 = dt_cbfs.online_CBF_parameters(state)

            sol, A_CBF, b_CBF, c_CBF, d_CBF = cbf_socp.solve(a_11, a_21, a_12, a_22, psi_m_r, psi_m_1, phi_m_r, phi_m_1,
                                                             psi_Lr_bar, phi_Lr_bar, psi_L1_bar, phi_L1_bar,
                                                             u_RL, u_bar)

            u_CBF = sol[0]
            print(f'u_CBF: {u_CBF:.5f}')

            ID = np.clip(u_bar + u_RL + u_CBF, a_min=min_action, a_max=max_action)  # pump limit
            control = np.array([ID, CHOs[i, 0]])

            if evaluate:
                sol_no_violations = sol.copy()
                sol_no_violations[-3] = 0.
                sol_no_violations[-2] = 0.

                states.append(state)
                controls.append(control)
                worst_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0] -
                                                np.linalg.norm(A_CBF[0] @ sol_no_violations + b_CBF[0])))
                worst_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1] -
                                                np.linalg.norm(A_CBF[1] @ sol_no_violations + b_CBF[1])))
                mean_CBF_psi.append(np.squeeze(c_CBF[0].T @ sol_no_violations + d_CBF[0]))
                mean_CBF_phi.append(np.squeeze(c_CBF[1].T @ sol_no_violations + d_CBF[1]))
                nominal_CBF_psi.append(a_11 * (sol_no_violations[0] + u_RL + u_bar) + a_21)
                nominal_CBF_phi.append(a_12 * (sol_no_violations[0] + u_RL + u_bar) + a_22)
                epsilon_psi.append(sol[-3])
                epsilon_phi.append(sol[-2])
                at_11, at_21, at_12, at_22 = dt_cbfs.online_true_CBF_parameters(state)
                true_CBF_psi.append(at_11 * (sol_no_violations[0] + u_RL + u_bar) + at_21)
                true_CBF_phi.append(at_12 * (sol_no_violations[0] + u_RL + u_bar) + at_22)
                u_RL_controls.append(u_RL)
                u_bar_controls.append(u_bar)
                u_CBF_controls.append(u_CBF)

            new_state = DT_Bergman_dynamics(state, control, true_model_params, dt)
            if new_state[0] < 50 or new_state[0] > 250:
                print('Try again! Blood glucose level went to an overly dangerous hyper/hypoglycemia level!')
                raise SystemExit

            reward = reward_calc(new_state[0])

            if not evaluate:
                state_action = np.concatenate((state, control[:1]), axis=0)  # action needs to be (1,)
                dt_cbfs.X_ht.add(state_action)

                if i >= dt_cbfs.r:
                    dt_cbfs.GP_collect_data(eating[i - dt_cbfs.r])

                if j <= compensator_tr_games and not restore_training:
                    compensator.collect_transition(state, u_bar + u_CBF)  # a Line Search algorithm is used to
                    # optimize the loss function over all collected data, so it means I cannot store anything and it's
                    # not done indeed in the OG implementation. I want to find immediately a good barrier for my RL
                    # agent so to let him train in a stable environment. So maximum precision with LIne Search for small
                    # dataset. Moreover, useless to start collecting again after restoring training and then train,
                    # you'll get the compensator just less precise. So ATTENTION when restoring training: assure the
                    # previous training had time to at least pass the last training episode for the compensator!

                agent.collect_transition(state, u_RL, reward, new_state, done=False)
                # done=True --> for transitions where the episode terminates by reaching some failure state,
                # and not due to the episode running until the max horizon (TD3 paper appendix)
                agent.learn()

            score += reward
            state = new_state
            # END EPISODE CYCLE

        if evaluate:
            print(f'Episode terminated. Score {score:.1f}. Max violation {max(episode_violations):.1f}.\n')
        else:
            max_violation_history.append(max(episode_violations))
            score_history.append(score)
            avg_score = np.mean(score_history[-reward_avg_window:])
            print(
                f'Episode {j} terminated. Score {score:.1f}. Avg score {avg_score:.1f}. Max violation {max(episode_violations):.1f}.\n')

            compensator_saved = False
            if j == compensator_tr_games and not restore_training:
                compensator.learn()
                compensator_saved = compensator.save_model()
                print(f"Is the compensator model saved? {compensator_saved}.\n")

            agent_saved = False
            if avg_score > best_score:
                best_score = avg_score
                agent_saved = agent.save_models()
                print(f"Is the agent model saved? {agent_saved}.\n")

            X, Y_psi, Y_phi = dt_cbfs.GP_memory.special_sample_buffer()

            _, idx = np.unique(X, axis=0, return_index=True)
            if len(idx) < X.shape[0]:
                print("Beware: in X there are", X.shape[0] - len(idx), "duplicated rows.")
            # print(f'X shape: {X.shape}, take a look: {X[:10, :]}\n')
            # print(f'Y_psi shape: {Y_psi.shape}, take a look: {Y_psi[:10, :]}\n')
            # print(f'Y_phi shape: {Y_phi.shape}, take a look: {Y_phi[:10, :]}\n')

            state_normalizer.adapt(X[:, :state_dim])

            X_ext = np.concatenate((state_normalizer(X[:, :state_dim]), np.ones((X.shape[0], 1)),
                                    X[:, state_dim:state_dim + action_dim]), axis=1)

            psi_model = gpflow.models.GPR((X_ext, Y_psi), kernel=kernels_psi)
            phi_model = gpflow.models.GPR((X_ext, Y_phi), kernel=kernels_phi)

            opt = gpflow.optimizers.Scipy()
            opt.minimize(psi_model.training_loss, psi_model.trainable_variables)
            opt.minimize(phi_model.training_loss, phi_model.trainable_variables)

            gpflow.utilities.print_summary(psi_model)
            gpflow.utilities.print_summary(phi_model)

            beta_rows, psi_m_right_factor, psi_L_hat = preliminary_computations_for_mnsrc(psi_model)
            _, phi_m_right_factor, phi_L_hat = preliminary_computations_for_mnsrc(phi_model)

            psi_model.state_normalizer = state_normalizer
            phi_model.state_normalizer = state_normalizer

            psi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=psi_model, beta_rows=beta_rows,
                                                                  m_right_factor=psi_m_right_factor, L_hat=psi_L_hat,
                                                                  action_dim=action_dim),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            phi_model.compiled_mean_and_square_root_covariance = tf.function(
                lambda x: compute_mean_and_square_root_covariance(x, model=phi_model, beta_rows=beta_rows,
                                                                  m_right_factor=phi_m_right_factor, L_hat=phi_L_hat,
                                                                  action_dim=action_dim),
                input_signature=[tf.TensorSpec(shape=[1, state_dim], dtype=tf.float64)],
            )

            tf.saved_model.save(psi_model, GP_psi_dir)
            tf.saved_model.save(phi_model, GP_phi_dir)
            if agent_saved:
                tf.saved_model.save(psi_model, GP_psi_dir + '/restored')
                tf.saved_model.save(phi_model, GP_phi_dir + '/restored')
        # END GAMES CYCLE

    if evaluate:
        plot_evaluation_run_with_GPs(dt, states, controls, worst_CBF_psi, worst_CBF_phi, mean_CBF_psi, mean_CBF_phi,
                                     nominal_CBF_psi, nominal_CBF_phi, epsilon_psi, epsilon_phi,
                                     k_delta, true_CBF_psi, true_CBF_phi,
                                     figure_file=f'plots/BergmanEvaluationRunTD3-Score{score:.0f}.png')
        with open('evaluation_run_details.txt', 'w') as f:
            for i in range(len(states)):
                f.write(f'Step {i}. Current BG level: {states[i][0]}.\nu_RL: {u_RL_controls[i]}\n'
                        f'u_bar: {u_bar_controls[i]}\nu_CBF: {u_CBF_controls[i]}\n'
                        f'Pre-pump u: {u_RL_controls[i] + u_bar_controls[i] + u_CBF_controls[i]}\n')
    else:
        x = [i + 1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, reward_avg_window, figure_file='plots/BergmanTrainingScoreTD3.png')
        plot_violation_curve(x, max_violation_history, figure_file='plots/BergmanTrainingViolationsTD3.png')
