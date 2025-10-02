import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
import math


class HistoryTracker:
    def __init__(self, past_number, time_steps, dimension):
        self.past_number = past_number
        self.time_steps = time_steps
        self.history = np.zeros((self.past_number+1, dimension))
        self.buffer = np.zeros((1, dimension))
        self.first_addition = True

    def add(self, measurement):
        if self.first_addition:
            self.buffer = self.buffer[1:]
        self.buffer = np.vstack((self.buffer, measurement))
        if self.buffer.shape[0] > self.past_number * self.time_steps + 1:
            self.buffer = self.buffer[1:]
        for i in range(self.buffer.shape[0]):
            if ((self.buffer.shape[0] - 1) - i) % self.time_steps == 0:
                self.history[int(((self.buffer.shape[0]-1)-i) / self.time_steps)] = self.buffer[i]
        self.first_addition = False


def DT_Bergman_dynamics(x, u, p, dt):
    p_1, G_b, p_2, p_3 = p['p_1'], p['G_b'], p['p_2'], p['p_3']
    n, I_b = p['n'], p['I_b']
    tau_G, V_G = p['tau_G'], p['V_G']

    G, X, I, D_2, D_1 = x[0], x[1], x[2], x[3], x[4]

    ID, CHO = u[0], u[1]

    G_next = G + dt * (- p_1 * (G - G_b) - G * X + (1 / (V_G * tau_G)) * D_2)
    X_next = X + dt * (- p_2 * X + p_3 * (I - I_b))
    I_next = I + dt * (- n * (I - I_b) + ID)
    D_2_next = D_2 + dt * (- D_2 / tau_G + D_1 / tau_G)
    D_1_next = D_1 + dt * (- D_1 / tau_G + CHO)

    return np.array([G_next, X_next, I_next, D_2_next, D_1_next])


def plot_reward_function(figure_file):
    g_series = range(50+5, 250-5)
    g_series = np.array(g_series)
    reward_series = np.zeros(len(g_series))
    for i, g in enumerate(g_series):
        reward_series[i] = reward_calc(g)
    plt.plot(g_series, reward_series)
    plt.axvspan(90, 180, facecolor='blue', alpha=0.25)
    plt.axvspan(180, 250-5, facecolor='red', alpha=0.25)
    plt.grid(True)
    plt.xlim(50+5, 250-5)
    max_reward = max(reward_series)
    min_reward = min(reward_series)
    plt.ylim(min_reward, max_reward + 1/40 * (max_reward - min_reward))
    plt.xlabel('G [mg/dl]')
    plt.ylabel('R(G)')
    plt.title('Reward function.')
    plt.savefig(figure_file)
    plt.close()


def reward_calc(bg):

    if 50 <= bg <= 90:
        reward = - 1.5

    elif 180 <= bg <= 250:
        reward = - 1

    elif 90 < bg < 180:
        if bg >= 140:
            reward = 1 + (-1 - 1) / (180 - 140) * (bg - 140)
        else:
            reward = 1 + (-1.5 - 1) / (90 - 140) * (bg - 140)

    return reward


def manage_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plot_learning_curve(x, scores, window_size, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-window_size):(i+1)])
    plt.plot(x, running_avg)
    plt.title(f'Running average of the previous {window_size} scores')
    plt.savefig(figure_file)
    plt.close()


def plot_violation_curve(x, violations, figure_file):
    plt.plot(x, violations)
    plt.title(f'Safety violations')
    plt.savefig(figure_file)
    plt.close()


def plot_evaluation_run_with_GPs(dt, states, controls, worst_CBF_psi, worst_CBF_phi, mean_CBF_psi,
                                 mean_CBF_phi, nominal_CBF_psi, nominal_CBF_phi,
                                 epsilon_psi, epsilon_phi, k_delta, true_CBF_psi, true_CBF_phi, figure_file):

    states = np.array(states)
    controls = np.array(controls)
    worst_CBF_psi = np.array(worst_CBF_psi)
    worst_CBF_phi = np.array(worst_CBF_phi)
    mean_CBF_psi = np.array(mean_CBF_psi)
    mean_CBF_phi = np.array(mean_CBF_phi)
    nominal_CBF_psi = np.array(nominal_CBF_psi)
    nominal_CBF_phi = np.array(nominal_CBF_phi)
    epsilon_psi = np.array(epsilon_psi)
    epsilon_phi = np.array(epsilon_phi)
    true_CBF_psi = np.array(true_CBF_psi)
    true_CBF_phi = np.array(true_CBF_phi)
    time_series = range(states.shape[0])

    fig, axs = plt.subplots(4, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 4, 1, 1]},
                            constrained_layout=True)

    color = 'tab:blue'
    axs[0].set_xlabel('t [min]')
    axs[0].set_ylabel('G(t) [mg/dl]', color=color)
    axs[0].plot(dt * np.array(time_series), states[:, 0], color=color)
    axs[0].axhline(y=180, color=color, linestyle='--')
    axs[0].axhline(y=90, color=color, linestyle='--')
    axs[0].tick_params(axis='y', labelcolor=color)

    ax12 = axs[0].twinx()

    color = 'tab:red'
    ax12.set_xlabel('t [min]')
    ax12.set_ylabel('ID(t) [$\mu$U/l/min]', color=color)
    markerline, stemlines, baseline = ax12.stem(
        dt * np.array(time_series), controls[:, 0], linefmt=color)
    alpha_val = 0.3
    markerline.set_alpha(alpha_val)
    stemlines.set_alpha(alpha_val)
    baseline.set_alpha(alpha_val)

    ax12.tick_params(axis='y', labelcolor=color)

    axs[0].grid(True)

    legend_labels = [
        'Mean',
        'Nominal',
        'True'
    ]
    legend_linestyles = ['-', '-.', ':', '--']
    color = 'tab:brown'
    axs[1].set_xlabel('t [min]')
    axs[1].set_ylabel('$\psi_{3}$(x(t)) [mg/dl]', color=color)
    axs[1].plot(dt * np.array(time_series), mean_CBF_psi, legend_linestyles[0], color=color, label=legend_labels[0])
    axs[1].fill_between(
        dt * np.array(time_series), worst_CBF_psi, mean_CBF_psi, color=color, alpha=0.25
    )
    axs[1].plot(dt * np.array(time_series), nominal_CBF_psi, legend_linestyles[1], color=color, label=legend_labels[1])
    axs[1].plot(dt * np.array(time_series), true_CBF_psi, legend_linestyles[2], color=color, label=legend_labels[2])
    axs[1].axhline(y=0, color=color, linestyle=legend_linestyles[3])
    axs[1].tick_params(axis='y', labelcolor=color)

    ax22 = axs[1].twinx()

    color = 'tab:purple'
    ax22.set_xlabel('t [min]')
    ax22.set_ylabel('$\phi_{3}$(x(t)) [mg/dl]', color=color)
    ax22.plot(dt * np.array(time_series), mean_CBF_phi, legend_linestyles[0], color=color)
    ax22.fill_between(
        dt * np.array(time_series), worst_CBF_phi, mean_CBF_phi, color=color, alpha=0.25
    )
    ax22.plot(dt * np.array(time_series), nominal_CBF_phi, legend_linestyles[1], color=color)
    ax22.plot(dt * np.array(time_series), true_CBF_phi, legend_linestyles[2], color=color)
    ax22.axhline(y=0, color=color, linestyle=legend_linestyles[3])
    ax22.tick_params(axis='y', labelcolor=color)

    axs[1].grid(True)
    custom_lines = []
    for i, label in enumerate(legend_labels):
        line = Line2D([0], [0], color='black', lw=2, ls=legend_linestyles[i])
        custom_lines.append(line)
    CBF_legend = axs[1].legend(custom_lines, legend_labels, loc='best', title=f'{k_delta} std of confidence')
    CBF_legend.get_title().set_fontsize('medium')
    CBF_legend.get_title().set_fontweight('bold')

    color = 'tab:brown'
    axs[2].set_xlabel('t [min]')
    axs[2].set_ylabel(r'$\epsilon_{\psi}$(t)', color=color)
    axs[2].plot(dt * np.array(time_series), epsilon_psi, color=color)  # CHECK VIOLATIONS ARRAY
    axs[2].tick_params(axis='y', labelcolor=color)

    ax32 = axs[2].twinx()

    color = 'tab:purple'
    ax32.set_xlabel('t [min]')
    ax32.set_ylabel(r'$\epsilon_{\phi}$(t)', color=color)
    ax32.plot(dt * np.array(time_series), epsilon_phi, color=color)  # CHECK VIOLATIONS ARRAY
    ax32.tick_params(axis='y', labelcolor=color)

    axs[2].grid(True)

    color = 'tab:green'
    axs[3].set_xlabel('t [min]')
    axs[3].set_ylabel('CHO(t) [mg/min]', color=color)
    axs[3].plot(dt * np.array(time_series), controls[:, 1], color=color)
    axs[3].tick_params(axis='y', labelcolor=color)

    axs[3].grid(True)

    plt.setp(markerline, visible=False)

    fig.suptitle('Single evaluation run')
    fig.savefig(figure_file, dpi=300)
    plt.close(fig)


def meal_schedule(N, dt, CHO_max):
    # # # MEAL SCHEDULE # # #
    # the meal protocol definition and generation algorithm should be thought in hours
    # hour | activity | CHO intake (g)
    # 7 wake up
    # 8+-0.15 breakfast 50+-10%
    # 11+-0.15 snack 15+-10%
    # 14+-0.15 lunch 65+-10%
    # 17+-0.15 snack 15+-10%
    # 20+-0.15 dinner 40+-10%
    CHOs = np.zeros((N, 1))
    hour = int(60/dt)
    n = divmod(N, 24 * hour)
    for i in range(n[0] + 1):
        meal_time = int(60 * (1 + np.random.choice([-0.15, 0.15]))/dt)
        if i != n[0] or n[1] >= meal_time and (i * 24 * hour + meal_time) <= N - 1:
            CHOs[i * 24 * hour + meal_time, 0] = np.random.choice([0.9, 1.1]) * 50_000 / dt
        meal_time = int(60 * (4 + np.random.choice([-0.15, 0.15]))/dt)
        if i != n[0] or n[1] >= meal_time and (i * 24 * hour + meal_time) <= N - 1:
            CHOs[i * 24 * hour + meal_time, 0] = np.random.choice([0.9, 1.1]) * 15_000 / dt
        meal_time = int(60 * (7 + np.random.choice([-0.15, 0.15]))/dt)
        if i != n[0] or n[1] >= meal_time and (i * 24 * hour + meal_time) <= N - 1:
            CHOs[i * 24 * hour + meal_time, 0] = np.random.choice([0.9, 1.1]) * 65_000 / dt
        meal_time = int(60 * (10 + np.random.choice([-0.15, 0.15]))/dt)
        if i != n[0] or n[1] >= meal_time and (i * 24 * hour + meal_time) <= N - 1:
            CHOs[i * 24 * hour + meal_time, 0] = np.random.choice([0.9, 1.1]) * 15_000 / dt
        meal_time = int(60 * (13 + np.random.choice([-0.15, 0.15]))/dt)
        if i != n[0] or n[1] >= meal_time and (i * 24 * hour + meal_time) <= N - 1:
            CHOs[i * 24 * hour + meal_time, 0] = np.random.choice([0.9, 1.1]) * 40_000 / dt
    CHOs = np.clip(CHOs, 0.0/dt, CHO_max/dt)
    eating = [bool(CHO != 0) for CHO in CHOs[:, 0]]
    return CHOs, eating


class ExplorationPolicy:
    def __init__(self, init_sigma, episode_length, final_sigma, action_tolerance, evaluation_episodes):
        self.init_sigma = init_sigma
        self.final_sigma = final_sigma
        self.action_tolerance = action_tolerance
        self.episode_length = episode_length
        self.evaluation_episodes = evaluation_episodes
        self.tau_fraction = 1 - ((final_sigma + action_tolerance)/init_sigma) ** (1/(episode_length -
                                                                                     self.evaluation_episodes))
        self.sigma = self.init_sigma
        self.n = 0
        if episode_length <= self.evaluation_episodes:
            raise ValueError(f'Episode length should be higher than {self.evaluation_episodes}, '
                             f'but is {self.episode_length}!')
        if (not (0 < (final_sigma + action_tolerance)/init_sigma < 1) or init_sigma <= 0 or final_sigma <= 0
                or action_tolerance <= 0):
            raise ValueError(f'Check the parameters of initial sigma, final sigma and the action tolerance!')

    def get_sigma(self):
        if self.n >= self.episode_length - self.evaluation_episodes:
            sigma = 0.
        else:
            sigma = self.sigma
            self.sigma = (1 - self.tau_fraction) * sigma
        self.n += 1
        return sigma

    def reset(self):
        self.sigma = self.init_sigma
        self.n = 0
