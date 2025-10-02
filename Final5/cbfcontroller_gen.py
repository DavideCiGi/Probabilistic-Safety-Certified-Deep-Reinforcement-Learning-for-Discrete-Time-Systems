import sympy as sp
import numpy as np
from utils import HistoryTracker
from buffer import GPBuffer
import cvxpy as cp
from cvxpygen import cpg

import math


class DT_CBFs:
    def __init__(self, dt, nominal_model_params, CBF1_params, CBF2_params, CHO_max, GP_max_size, episode_length,
                 true_model_params):

        r = 3  # relative degree
        BG_max_value = 180
        BG_min_value = 90

        BG_min = sp.symbols('BG_{min}', real=True)
        BG_max = sp.symbols('BG_{max}', real=True)

        G_temp = [sp.symbols(f"G({i})", real=True) for i in range(r + 1)]

        h_1 = [g - BG_min for g in G_temp]
        h_2 = [- g + BG_max for g in G_temp]
        # barrier function evaluated at subsequent time steps

        # Now I will just build a matrix psi in order to get the expression of the
        # DT H-O CBF. It's just a matrix with no real meaning (for now)
        psi = sp.Matrix([sp.symbols(f'psi_{i}{j}', real=True)
                         for i in range(r + 1) for j in range(r + 1)], real=True).reshape(r + 1, r + 1)
        phi = sp.Matrix([sp.symbols(f'phi_{i}{j}', real=True)
                         for i in range(r + 1) for j in range(r + 1)], real=True).reshape(r + 1, r + 1)

        for j in range(r + 1):
            psi[0, j] = h_1[j]
            phi[0, j] = h_2[j]

        gamma_1 = [sp.Symbol(f'\gamma_{{{i + 1}, 1}}', real=True) for i in range(r)]
        gamma_2 = [sp.Symbol(f'\gamma_{{{i + 1}, 2}}', real=True) for i in range(r)]

        for i in range(r):
            for j in range(r - i):
                psi[i + 1, j] = psi[i, j + 1] - psi[i, j] + gamma_1[i] * psi[i, j]
                phi[i + 1, j] = phi[i, j + 1] - phi[i, j] + gamma_2[i] * phi[i, j]
        # Now let's retrieve the actual symbolic expressions for the psi functions!

        # What you actually retrieve is the series of functions built over the cbf
        # but only as a linear combination of the cbf evaluated up till the r step
        psi_r = psi[-1, 0].copy()
        phi_r = phi[-1, 0].copy()

        psi_actual = psi[1:, 0].copy()
        phi_actual = phi[1:, 0].copy()

        delta = sp.Symbol(r'\Delta t', real=True)  # sampling time
        p_1, G_b, p_2, p_3, n, I_b, tau_G, V_G = sp.symbols(
            'p_1, G_b, p_2, p_3, n, I_b, tau_G, V_G', real=True)

        G = [sp.symbols(f"G({i})", real=True) for i in range(r + 1)]
        X = [sp.symbols(f"X({i})", real=True) for i in range(r + 1)]
        I = [sp.symbols(f"I({i})", real=True) for i in range(r + 1)]
        D_2 = [sp.symbols(f"D_{{2}}({i})", real=True) for i in range(r + 1)]
        D_1 = [sp.symbols(f"D_{{1}}({i})", real=True) for i in range(r + 1)]
        state_dim = 5

        ID = [sp.symbols(f"ID({i})", real=True) for i in range(r + 1)]
        action_dim = 1
        CHO = [sp.symbols(f"CHO({i})", real=True) for i in range(r + 1)]

        for i in range(r):
            G[i + 1] = G[i] + delta * (- p_1 * (G[i] - G_b) - G[i] * X[i] + (1 / (V_G * tau_G)) * D_2[i])
            X[i + 1] = X[i] + delta * (- p_2 * X[i] + p_3 * (I[i] - I_b))
            I[i + 1] = I[i] + delta * (- n * (I[i] - I_b) + ID[i])
            D_2[i + 1] = D_2[i] + delta * (- D_2[i] / tau_G + D_1[i] / tau_G)
            D_1[i + 1] = D_1[i] + delta * (- D_1[i] / tau_G + CHO[i])

        psi_r = psi_r.subs(dict(zip(G_temp, G)))  # CHECK
        phi_r = phi_r.subs(dict(zip(G_temp, G)))

        psi_true_r = psi_r.copy()
        phi_true_r = phi_r.copy()

        nominal_model_params = {
            p_1: nominal_model_params['p_1'],
            G_b: nominal_model_params['G_b'],
            p_2: nominal_model_params['p_2'],
            p_3: nominal_model_params['p_3'],
            n: nominal_model_params['n'],
            I_b: nominal_model_params['I_b'],
            tau_G: nominal_model_params['tau_G'],
            V_G: nominal_model_params['V_G']
        }

        true_model_params = {
            p_1: true_model_params['p_1'],
            G_b: true_model_params['G_b'],
            p_2: true_model_params['p_2'],
            p_3: true_model_params['p_3'],
            n: true_model_params['n'],
            I_b: true_model_params['I_b'],
            tau_G: true_model_params['tau_G'],
            V_G: true_model_params['V_G']
        }

        shared_params = {BG_min: BG_min_value, BG_max: BG_max_value, delta: dt}

        nominal_model_params.update(shared_params)
        true_model_params.update(shared_params)

        CBF1_params = {
            gamma_1[0]: CBF1_params['gamma_11'],
            gamma_1[1]: CBF1_params['gamma_21'],
            gamma_1[2]: CBF1_params['gamma_31']
        }

        CBF2_params = {
            gamma_2[0]: CBF2_params['gamma_12'],
            gamma_2[1]: CBF2_params['gamma_22'],
            gamma_2[2]: CBF2_params['gamma_32']
        }

        if not (len(CBF1_params) == len(CBF2_params)) & (len(CBF1_params) == r):  # it is necessary because
            # in the main loop, in the first training step, I am assuming len(CBF_1_params) == dt_cbfs.r! Since SOCP
            # utilizes len(CBF_1_params) to get the relative degree!
            raise ValueError(f'The number of auxiliary functions parameters {len(CBF1_params)}'
                             f' and {len(CBF2_params)} does not coincide with the input relative degree {r}!')

        psi_r = psi_r.subs(nominal_model_params).subs(CBF1_params)
        phi_r = phi_r.subs(nominal_model_params).subs(CBF2_params)

        psi_true_r = psi_true_r.subs(true_model_params).subs(
            CBF1_params)  # of course we shouldn't have access to the true
        # parameters. This is just a way to double check if what we're doing is ok or not. You can even eliminate it
        phi_true_r = phi_true_r.subs(true_model_params).subs(CBF2_params)

        psi_actual = psi_actual.subs(shared_params).subs(CBF1_params)
        phi_actual = phi_actual.subs(shared_params).subs(CBF2_params)

        # let's implement robustness, CHO[0] intervenes at psi[r], phi[r], so it
        # doesn't intervene the initial check of the CBFs.
        psi_r = psi_r.subs({CHO[0]: 0})
        phi0_r = phi_r.subs({CHO[0]: 0})  # function we'll use later for GPs
        phi_r = phi_r.subs({CHO[0]: CHO_max})

        psi_true_r = psi_true_r.subs({CHO[0]: 0})
        phi_true_r = phi_true_r.subs({CHO[0]: CHO_max})

        psi_r_function = sp.lambdify(
            (G[0], X[0], I[0], D_2[0], D_1[0], ID[0]), psi_r, modules='numpy')
        phi_r_function = sp.lambdify(
            (G[0], X[0], I[0], D_2[0], D_1[0], ID[0]), phi_r, modules='numpy')
        phi0_r_function = sp.lambdify(
            (G[0], X[0], I[0], D_2[0], D_1[0], ID[0]), phi0_r, modules='numpy')
        psi_true_r_function = sp.lambdify(
            (G[0], X[0], I[0], D_2[0], D_1[0], ID[0]), psi_true_r, modules='numpy')
        phi_true_r_function = sp.lambdify(
            (G[0], X[0], I[0], D_2[0], D_1[0], ID[0]), phi_true_r, modules='numpy')

        self.psi_r_function = psi_r_function
        self.phi_r_function = phi_r_function
        self.psi_true_r_function = psi_true_r_function
        self.phi_true_r_function = phi_true_r_function
        self.phi0_r_function = phi0_r_function

        self.r = r

        self.psi_actual = psi_actual
        self.phi_actual = phi_actual
        self.G_temp = G_temp

        self.dt = dt

        self.GP_memory = GPBuffer(GP_max_size, state_dim, action_dim, episode_length)
        self.X_ht = HistoryTracker(r, 1, state_dim + action_dim)

        self.BG_max_value = BG_max_value
        self.BG_min_value = BG_min_value
        

    def online_CBF_parameters(self, x0):
        a_21 = self.psi_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_11 = self.psi_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_21

        a_22 = self.phi_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        a_12 = self.phi_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - a_22

        return a_11, a_21, a_12, a_22

    def online_true_CBF_parameters(self, x0):
        at_21 = self.psi_true_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        at_11 = self.psi_true_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - at_21

        at_22 = self.phi_true_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 0.0)
        at_12 = self.phi_true_r_function(x0[0], x0[1], x0[2], x0[3], x0[4], 1.0) - at_22

        return at_11, at_21, at_12, at_22

    def GP_collect_data(self, eating):
        if not eating:
            previous_state_action = self.X_ht.history[-1]  # old action, 3 steps before the current one
            psi_error = self.psi_actual[-1].copy()
            phi_error = self.phi_actual[-1].copy()
            psi_estimate = (self.psi_r_function(*previous_state_action)).copy()
            phi_estimate = (self.phi0_r_function(*previous_state_action)).copy()
            for k in range(self.r + 1):
                psi_error = psi_error.subs({self.G_temp[k]: self.X_ht.history[-(k + 1), 0]})
                phi_error = phi_error.subs({self.G_temp[k]: self.X_ht.history[-(k + 1), 0]})
            psi_error -= psi_estimate
            phi_error -= phi_estimate
            self.GP_memory.store(previous_state_action, psi_error, phi_error)


class CBF_with_GPs_SOCP:
    def __init__(self, k_delta, K_eps_1, K_eps_2, max_action, min_action, gamma_1_dict, gamma_2_dict):
        self.k_delta = k_delta
        self.min_action = min_action
        self.max_action = max_action
        gamma_1 = list(gamma_1_dict.values())
        gamma_2 = list(gamma_2_dict.values())

        r = 1

        m = 4
        n = 4
        f = cp.Constant(np.array([0., 0., 0., 1.]))

        A_1 = cp.Constant(np.diag([1., K_eps_1, K_eps_2, 0.]))
        A_2 = cp.Constant(np.hstack((np.array([[2.]]), np.zeros((1, 3)))))
        A_31 = cp.Parameter(shape=(r + 1, 1), name='A_31')
        A_32 = cp.Constant(np.zeros((r + 1, 3)))
        A_3 = cp.hstack([A_31, A_32])
        A_41 = cp.Parameter(shape=(r + 1, 1), name='A_41')
        A_42 = cp.Constant(np.zeros((r + 1, 3)))
        A_4 = cp.hstack([A_41, A_42])

        A = [A_1, A_2, A_3, A_4]

        b_1 = cp.Constant(np.zeros(1))
        b_2 = cp.Parameter(shape=(1,), name='b_2')  # CHECK???
        b_3 = cp.Parameter(shape=(r + 1,), name='b_3')
        b_4 = cp.Parameter(shape=(r + 1,), name='b_4')

        b = [b_1, b_2, b_3, b_4]

        c_1 = cp.Constant(np.concatenate((np.zeros(3), np.array([1.]))))
        c_2 = cp.Constant(np.zeros(n))
        c_31 = cp.Parameter(name="c_31")
        c_32 = cp.Constant(np.concatenate((np.array([math.prod(gamma_1)]), np.zeros(2))))
        c_3 = cp.hstack([c_31, c_32])
        c_41 = cp.Parameter(name="c_41")
        c_42 = cp.Constant(np.concatenate((np.zeros(1), np.array([math.prod(gamma_2)]), np.zeros(1))))
        c_4 = cp.hstack([c_41, c_42])

        c = [c_1, c_2, c_3, c_4]

        d_1 = cp.Constant(np.zeros(1))
        d_2 = cp.Constant(np.array([(self.max_action - self.min_action)]))
        d_3 = cp.Parameter(shape=(1,), name='d_3')
        d_4 = cp.Parameter(shape=(1,), name='d_4')

        d = [d_1, d_2, d_3, d_4]

        # Define and solve the CVXPY problem.
        x = cp.Variable(n, name='x')

        # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
        soc_constraints = [
            cp.SOC(c[j].T @ x + d[j], A[j] @ x + b[j]) for j in range(m)
        ]

        l_constraints = [
            x[-3] >= 0.,
            x[-2] >= 0.
        ]

        self.prob = cp.Problem(cp.Minimize(f.T @ x),
                               soc_constraints + l_constraints)
        # print(self.prob)
        # print("Is DPP? ", self.prob.is_dcp(dpp=True))
        # print("Is DCP? ", self.prob.is_dcp(dpp=False))

        A_31.value = np.zeros((r + 1, 1))
        A_41.value = np.zeros((r + 1, 1))
        b_2.value = np.random.rand(1)
        b_3.value = np.random.rand(r + 1)
        b_4.value = np.random.rand(r + 1)
        c_31.value = np.random.rand()
        c_41.value = np.random.rand()
        d_3.value = np.random.rand(1)
        d_4.value = np.random.rand(1)

        """
        # CLARABEL

        self.prob.solve(solver='CLARABEL')

        cpg.generate_code(self.prob, code_dir='SOCP_code', solver='CLARABEL', enable_settings=['verbose'])
        """

        # ECOS

        self.prob.solve(solver='ECOS')

        cpg.generate_code(self.prob, code_dir='SOCP_code', solver='ECOS')

        from SOCP_code.cpg_solver import cpg_solve
        self.cpg_solve = cpg_solve

        self.A_32 = A_32
        self.A_42 = A_42
        self.c_32 = c_32
        self.c_42 = c_42

    def solve(self, a_11, a_21, a_12, a_22, psi_m_r, psi_m_1, phi_m_r, phi_m_1, psi_Lr_bar, phi_Lr_bar,
              psi_L1_bar, phi_L1_bar, u_RL, u_bar):
        A_31 = self.k_delta * psi_L1_bar
        A_41 = self.k_delta * phi_L1_bar
        b_2 = np.array([2 * (u_RL + u_bar) - (self.max_action + self.min_action)])
        b_3 = np.squeeze(self.k_delta * (psi_Lr_bar @ np.array([[0.]]) + psi_L1_bar * (u_RL + u_bar)))
        b_4 = np.squeeze(self.k_delta * (phi_Lr_bar @ np.array([[0.]]) + phi_L1_bar * (u_RL + u_bar)))
        c_31 = a_11 + np.squeeze(psi_m_1)
        c_41 = a_12 + np.squeeze(phi_m_1)
        d_3 = np.array([a_21 + np.squeeze(psi_m_r.T @ np.array([[0.]])) + (a_11 + np.squeeze(psi_m_1)) * (u_RL + u_bar)])
        d_4 = np.array([a_22 + np.squeeze(phi_m_r.T @ np.array([[0.]])) + (a_12 + np.squeeze(phi_m_1)) * (u_RL + u_bar)])

        self.prob.param_dict['A_31'].value = A_31
        self.prob.param_dict['A_41'].value = A_41
        self.prob.param_dict['b_2'].value = b_2
        self.prob.param_dict['b_3'].value = b_3
        self.prob.param_dict['b_4'].value = b_4
        self.prob.param_dict['c_31'].value = c_31
        self.prob.param_dict['c_41'].value = c_41
        self.prob.param_dict['d_3'].value = d_3
        self.prob.param_dict['d_4'].value = d_4

        self.prob.register_solve('CPG', self.cpg_solve)

        """
        # CLARABEL

        self.prob.solve(method='CPG', updated_params=['A_31', 'A_41', 'b_2', 'b_3', 'b_4', 'c_31', 'c_41',
                                                      'd_3', 'd_4'], verbose=False)
        """

        # ECOS

        self.prob.solve(method='CPG', updated_params=['A_31', 'A_41', 'b_2', 'b_3', 'b_4', 'c_31', 'c_41',
                                                      'd_3', 'd_4'])

        x_opt = self.prob.var_dict['x'].value

        A_3 = np.hstack([A_31, self.A_32.value])
        A_4 = np.hstack([A_41, self.A_42.value])
        c_3 = np.hstack([c_31, self.c_32.value])
        c_4 = np.hstack([c_41, self.c_42.value])

        A_CBF = [A_3, A_4]
        b_CBF = [b_3, b_4]
        c_CBF = [c_3, c_4]
        d_CBF = [d_3, d_4]

        return x_opt, A_CBF, b_CBF, c_CBF, d_CBF
