import numpy as np
from matplotlib import pyplot as plt
from typing import List
import matplotlib as mpl
from ptlpinns.perturbation import standard
from ptlpinns.odes import equations, numerical
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import torch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

def relative_error(values: List[np.ndarray]) -> np.ndarray:
    """
    Plots the relative frequency error for a list of values.
    Can be used to check if LPM converged.
    """
    cmap = mpl.cm.get_cmap('viridis')
    color1 = cmap(0.2)

    r_error = np.abs(np.diff(values))
    steps = np.arange(1, len(r_error) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(
        steps, r_error, marker="o", markersize=7, linestyle="None",
        color=color1, linewidth=2, label="Frequency Correction"
    )
    plt.xlabel("Correction order", fontsize=16)
    plt.ylabel(r"|$\Delta \omega$|", fontsize=18)
    plt.yscale("log")

    plt.xticks(steps, fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.show()

def w_absolute_error(w_series: List[np.ndarray], w_teor: float) -> np.ndarray:
    """
    Plots the absolute frequency error for a given theoretical value.
    """
    cmap = mpl.cm.get_cmap('viridis')
    color2 = cmap(0.8)

    frequency_error = np.abs(w_series - w_teor)
    steps = np.arange(0, len(frequency_error))

    plt.figure(figsize=(8, 4))
    plt.plot(
        steps, frequency_error, marker="o", markersize=7, 
        linestyle="None", color=color2, linewidth=2
    )
    plt.xlabel("Correction order", fontsize=16)
    plt.ylabel(r"MAE($\omega$)", fontsize=18)
    plt.yscale("log")

    plt.xticks(steps, fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.show()

def epsilon_x_power(N: int, x: List, power):
    """
    Calculates the order-N contribution of ε * (x ** q)

    Parameters:
        N (int): the perturbation order
        x (List[np.ndarray]): list of x_0, x_1, ..., x_{N-1}
        power (int): the power of x in the nonlinearity (e.g., 3 for x³, 5 for x⁵)

    Returns:
        np.ndarray: ε * x^power term at order N
    """

    lib = np if type(x[0]) == np.ndarray else torch

    for nonlinearity in power:

        index_list = standard.index_tuples(N, nonlinearity[0])
        result = lib.zeros_like(x[0])
        for indices in index_list:
            term = standard.number_combinations(indices)
            for i in indices:
                term *= x[i]
            result += nonlinearity[1] * term

    return result

def calc_A(N: int, x_ddot: List, w_list: List)-> List:
    """
    Calculates A(x_{0}, ..., x_{N-1}, w_{0}, ..., w_{N - 1})

    Returns
        np.ndarray: Numpy array with A's values
    """

    lib = np if type(x_ddot[0]) == np.ndarray else torch
    A_term = lib.zeros_like(x_ddot[0])
    for k in range(1, N + 1): # ddot_x_N can't be calculated
        x_ddot_nk = x_ddot[N - k]
        for i in range(k + 1):
            if i == N or (k - i) == N: # 2 w_0 * w_N * ddot x_0 can't be calculated
                continue
            A_term += x_ddot_nk * w_list[i] * w_list[k - i] 

    return A_term

def calc_B(N: int, x_ddot: List, x: List, w_list: List, power = [(3, 1)]):
    epsilon_cubed_term = - epsilon_x_power(N, x, power)
    A_term = - calc_A(N, x_ddot, w_list)
    return A_term + epsilon_cubed_term

def calc_w_n(w_list: List, x:List, x_ddot: List, t, power = [(3, 1)]) -> int:
    """
    Calcultes the frequency correction for order n using quantities of order n - 1

    Returns:
        int: Frequency correction for order n
    """

    lib = np if type(x[0]) == np.ndarray else torch

    mask    = (t >= 0) & (t <= 2*lib.pi)
    t_seg   = t[mask]    

    x_seg = [term[mask] for term in x]
    x_ddot_seg = [term[mask] for term in x_ddot] 


    B = calc_B(N=len(x_ddot), x_ddot = x_ddot_seg, x = x_seg, w_list=w_list, power=power)
    K = 2 * x_ddot_seg[0] / w_list[0]

    num = lib.trapezoid(B * (lib.cos(t_seg)), x=t_seg)
    den = lib.trapezoid(K * (lib.cos(t_seg)), x=t_seg)

    w_n = num / den

    return w_n

def calculate_forcing(w_n: float, w_list: List, x:List, x_ddot: List, power = 3) -> np.ndarray:
    """"
    Calculates forcing for order n differential equation. Depends only on x and w from order 0 to n - 1.
    """

    B_term = calc_B(len(x_ddot), x_ddot, x, w_list, power)
    w_n_term = - 2 * w_n * x_ddot[0] / w_list[0]

    return w_n_term + B_term

def calculate_w_series(values: List[float], epsilon: float, rwtol = 1e-6, check_divergence = True) -> List[np.ndarray]:
    """
    For an array of values with `n` corrections (x or w or x') returns a List
    where index `n` is the series up to `n`

    Returns:
        List[np.ndarray]: List of the series corrections ranging from `0` to `n` order
    """
    w_0 = values[0]

    solution = w_0
    series = [w_0]
    last_delta = 3 * w_0 

    for i in range(1, len(values)):
        solution_old = solution
        solution += (epsilon ** i) * values[i] 
        new_delta = np.abs(solution - solution_old)

        if (new_delta > last_delta and check_divergence):
            print(f"series has diverged for order p = {i}")
            break

        if new_delta < rwtol:
            print(f"series has converged for order p = {i - 1} under tolerance {rwtol}")
            break
        
        last_delta = new_delta 
        series.append(solution)

    return series

def estimate_period_frequency(w_0, zeta, ic, q, epsilon, t_eval = np.linspace(0, 4 * np.pi, 10000)):
    """
    Can be used to estimate the period and frequency of a periodic signal

    Returns:
        T_avg (float): Average period of the signal
        omega (float): Frequency of the signal
    """

    ode = equations.ode_oscillator_1D(w_0=w_0, zeta=zeta, forcing_1D=lambda t: np.zeros_like(t), q=q, epsilon=epsilon)
    x_t = numerical.solve_ode_equation(ode, (t_eval[0], t_eval[-1]), t_eval, ic)[0]

    peak_indices, _ = find_peaks(x_t)
    
    if len(peak_indices) < 2:
        raise ValueError("Not enough peaks to estimate the period.")

    # Get corresponding times
    peak_times = t_eval[peak_indices]

    # Compute differences between consecutive peaks
    periods = np.diff(peak_times)
    T_avg = np.mean(periods)
    omega = 2 * np.pi / T_avg

    return T_avg, omega

def t_eval_lpm(t_eval, w_final):
    """
    Returns:
        t_eval_lpm: t_eval scaled for LPM solution
        t_eval_standard: t_eval clipped to the LPM solution range
    """
    t_eval_lpm = t_eval / w_final
    mask = (t_eval >= 0) & (t_eval <= t_eval_lpm[-1])
    t_eval_standard = t_eval[mask]

    return t_eval_lpm, t_eval_standard

def plot_multiple_phase_diagrams(
    arr_list, labels=None,
    xlab="position", ylab="velocity", title="Phase Diagram",
    s=8, alpha=0.6, xlim=None, ylim=None,
    lpm_index=None, omega=None, lpm_stride=30
    ):
    """
    Expects arrays in this order: [LPM, Numerical, Standard].

    Plot multiple phase diagrams on the same plot.
    Omega scales LPM velocity to ensure correct scaling. 
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

    cmap = plt.cm.viridis
    num_color = cmap(0.2)   # deep bluish
    std_color = cmap(0.5)   # medium greenish
    lpm_color = cmap(0.8)   # yellow-green

    if labels is None:
        labels = [f"Array {i+1}" for i in range(len(arr_list))]

    handles_all, labels_all = [], []

    order = [2, 1, 0]

    for i in order:
        arr = np.asarray(arr_list[i])
        label = labels[i]

        # Ensure shape (N, 2)
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T

        # Apply frequency scaling to the LPM solution
        if i == lpm_index and omega is not None:
            arr = arr.copy()
            arr[:, 1] *= omega

        mask = np.all(np.isfinite(arr), axis=1)
        x, y = arr[mask, 0], arr[mask, 1]

        if i == 2:  # Standard (back)
            h = ax.scatter(x, y, s=s, alpha=alpha, c=[std_color],
                           marker='.', label=label, zorder=1)
        elif i == 1:  # Numerical
            h, = ax.plot(x, y, color=num_color, linewidth=1.4,
                         label=label, zorder=2)
        elif i == 0:  # LPM (front, sparse points)
            idx = np.arange(0, len(x), max(1, lpm_stride))
            h = ax.scatter(x[idx], y[idx], s=s*2, alpha=1.0, c=[lpm_color],
                           marker='o', edgecolors='none', label=label, zorder=3)

        handles_all.append(h)
        labels_all.append(label)

    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    fig.legend(handles_all, labels_all,
               loc='upper center', ncol=3, frameon=False, fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

def plot_KG_solution(sol, c, t_eval, x_span, t_span, title="", w_lpm = 1):
    """
    Plots the solution of the Klein-Gordon equation in a 2D phase diagram
    from a 1D ODE solution.
    """

    t_eval = t_eval / w_lpm

    qsi_grid = np.linspace(t_eval[0], t_eval[-1], len(sol))
    interp_fun = interp1d(qsi_grid, sol, kind='linear', bounds_error=False, fill_value=np.nan)

    xmin, xmax = x_span
    tmin, tmax = t_span

    tmin = tmin
    tmax = tmax

    x = np.linspace(xmin, xmax, 1000)
    t = np.linspace(tmin, tmax, 1000)

    X, T = np.meshgrid(x, t)
    QSI = X - c * T
    D = interp_fun(QSI)

    plt.figure(figsize=(8,6))
    plt.pcolormesh(X, T, D, shading='auto', cmap='viridis')
    cbar = plt.colorbar()
    cbar.set_label(r'$u(\xi = x - ct)$', labelpad=8, fontsize=14)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title, fontsize=14, pad = 10)
    plt.show()


def plot_compare_1st_2nd_pass(numerical_1st_pass, numerical_2nd_pass, NN_TL_solution_LPM_1st_pass, NN_TL_solution_LPM_2nd_pass, t_eval_1st_pass, t_eval_2nd_pass):

    cmap = mpl.cm.get_cmap('viridis')
    color1 = cmap(0.2) 
    color2 = cmap(0.8) 

    plt.figure(figsize=(12, 4))

    plt.plot(
        t_eval_1st_pass, np.abs(numerical_1st_pass[0, :] - NN_TL_solution_LPM_1st_pass[:, 0]),
        label=r'MAE (PINN): 1st-pass',
        linewidth=2,
        color=color1,
        linestyle='-'
    )

    plt.plot(
        t_eval_2nd_pass, np.abs(numerical_2nd_pass[0, :] - NN_TL_solution_LPM_2nd_pass[:, 0]),
        label=r'MAE (PINN): 2nd-pass',
        linewidth=2,
        color=color2,
    )

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('MAE', fontsize=14)

    # Legend: above plot, centered
    plt.legend(
        fontsize=16,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


    plt.tight_layout()
    plt.show()