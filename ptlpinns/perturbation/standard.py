import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.integrate import cumulative_trapezoid
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
from typing import List
from collections import Counter
from math import factorial
from itertools import combinations_with_replacement
import matplotlib as mpl

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

def number_combinations(index_list: List[int]) -> int:
    """
    From a list with indices, calculate the number of unique combinations

    Args:
        index_list (List[int]): A list of indices to select terms from x 
        that will appear in forcing function.

    Returns:
        int: The number of unique combinations of indices in the list.

    Example:
        >>> number_combinations([0, 0, 1])
        3
        >>> number_combinations([0, 2, 2])
        3
        >>> number_combinations([1, 2, 3])
        6
        >>> number_combinations([0, 0, 0])
        1
        >>> number_combinations([0, 0, 0, 0, 0])
        1
        >>> number_combinations([0, 0, 0, 1, 1])
        10
        >>> number_combinations([0, 1, 2, 3, 4])
        120
    """
    counts = Counter(index_list)
    denom = 1
    for count in counts.values():
        denom *= factorial(count)

    return factorial(len(index_list)) // denom


def index_tuples(N: int, num_indices: int) -> List[List[int]]:
    """
    Generate all index tuples [i1, i2, ..., in] of length num_indices such that:
    - i1 <= i2 <= ... <= in
    - sum(i1, i2, ..., in) == N - 1

    Args:
        N (int): The target sum is N - 1
        num_indices (int): The number of indices in each tuple

    Returns:
        List[List[int]]: A list of index tuples satisfying the constraints

    Examples:
        >>> index_tuples(3, 2)
        [[0, 2], [1, 1]]
        >>> index_tuples(5, 3)
        [[0, 0, 4], [0, 1, 3], [0, 2, 2], [1, 1, 2]]
    """
    return [list(tup) for tup in combinations_with_replacement(range(N), num_indices)
            if sum(tup) == N - 1]

def force_func_perturbation(n, num_indices=3):

    index_list = index_tuples(n, num_indices)
    solution_index = [index_list[i] + [number_combinations(index_list[i])] for i in range(len(index_list))]
    return solution_index

def calculate_general_series(values: List, epsilon: float) -> List[np.ndarray]:
    """
    For an array of values with `n` corrections (x or w or x') returns a List
    where index `n` is the series up to `n`

    Returns:
        List[np.ndarray]: List of the series corrections ranging from `0` to `n` order
    """
    def safe_copy(x):
        return x.copy() if hasattr(x, 'copy') else x  # fall back for scalars

    solution = safe_copy(values[0])
    series = [safe_copy(solution)]

    for i in range(1, len(values)):
        solution += (epsilon ** i) * values[i]
        series.append(safe_copy(solution))

    return series

def plot_IAE_and_subplots(PINN_x_solution_series, numerical_undamped_duffing,
                           perturbation_solution, t_eval, selected_orders=[0, 1, 6], ylim=(0, 25)):

    colors = cm.viridis(np.linspace(0, 1, len(selected_orders)))
    linestyles = ['--' if i == 0 else '-' for i in selected_orders]

    fig1, ax1 = plt.subplots(figsize=(16, 4.5))

    lines = []
    for idx, i in enumerate(selected_orders):
        error = abs(PINN_x_solution_series[i] - numerical_undamped_duffing[0])
        iae = cumulative_trapezoid(error, t_eval, initial=0)
        line, = ax1.plot(
            t_eval,
            iae,
            label=f"Up to Order {i}",
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=2
        )
        lines.append(line)

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("Time", fontsize=16)
    ax1.set_ylabel("IAE(t)", fontsize=16)

    # Legend at the top
    fig1.legend(
        handles=lines,
        labels=[f"Up to order {i}" for i in selected_orders],
        fontsize=18,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=len(selected_orders),
        frameon=False
    )

    plt.ylim(ylim)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top
    plt.show()

    titles = [r"$x_0(t)$", r"$x_1(t)$", r"$x_6(t)$"]

    fig2, axes = plt.subplots(1, 3, figsize=(16, 3.5), sharey=False, constrained_layout=True)

    for i, order in enumerate(selected_orders):
        axes[i].plot(t_eval, perturbation_solution[order][:, 0], linewidth=2)
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].tick_params(axis='both', labelsize=14)
        axes[i].set_xlabel("Time", fontsize=14)

    axes[0].set_ylabel("Amplitude", fontsize=14)

    plt.show()

def epsilon_x_power(N: int, x: List[np.ndarray], power: int) -> np.ndarray:
    """
    Calculates the order-N contribution of ε * (x ** q)

    Parameters:
        N (int): the perturbation order
        x (List[np.ndarray]): list of x_0, x_1, ..., x_{N-1}
        power (int): the power of x in the nonlinearity (e.g., 3 for x³, 5 for x⁵)

    Returns:
        np.ndarray: ε * x^power term at order N
    """
    index_list = index_tuples(N, power)
    result = np.zeros_like(x[0])
    for indices in index_list:
        term = number_combinations(indices)
        for i in indices:
            term *= x[i]
        result += term
    return result



def plot_comparison_standard_vs_lpm(t_eval, t_eval_lpm, perturbation_solution_standard, NN_TL_solution_standard,
                                    perturbation_solution, NN_TL_solution, numerical_undamped_duffing, order=8):

    cmap = mpl.cm.get_cmap('viridis')
    num_color = cmap(0.2)   # deep bluish
    std_color = cmap(0.5)   # medium greenish
    lpm_color = cmap(0.8)  # yellow-green

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # --- right panel: Full solution up to 8th order ---
    axes[1].plot(t_eval, NN_TL_solution_standard[:, 0],
                label='PTL-PINN: standard perturbation', linestyle='--',
                color=std_color, linewidth=2)
    axes[1].plot(t_eval, numerical_undamped_duffing[0],
                label='Numerical', color=num_color, linewidth=2.2)
    axes[1].plot(t_eval_lpm, NN_TL_solution[:, 0],
                label='PTL-PINN: Lindstedt-Poincare method', linestyle=':',
                color=lpm_color, linewidth=2, marker='o', markevery=200, markersize=5)

    axes[0].set_ylim(-2, 2)
    axes[1].set_title(f"Up to {order}th-order", fontsize=16, pad=12)
    axes[1].set_xlabel("Time", fontsize=14)
    axes[0].set_ylabel("Amplitude", fontsize=14)

    # --- left panel: 0th-order solution ---
    axes[0].plot(t_eval, perturbation_solution_standard[0][:, 0],
                label='PTL-PINN: Standard', linestyle='--',
                color=std_color, linewidth=2)
    axes[0].plot(t_eval, numerical_undamped_duffing[0],
                label='Numerical', color=num_color, linewidth=2.2)
    axes[0].plot(t_eval_lpm, perturbation_solution[0][:, 0],
                label='PTL-PINN: Lindstedt-Poincare', linestyle=':',
                color=lpm_color, linewidth=2, marker='o', markevery=200, markersize=5)

    axes[0].set_ylim(-2, 2)
    axes[0].set_title("0th-order solution", fontsize=16, pad=12)
    axes[0].set_xlabel("Time", fontsize=14)

    # --- Tick settings and styles ---
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        ax.grid(True, linestyle=':', alpha=0.6)  # light grid
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # --- Legend on top ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=16)

    # --- Layout adjustments ---
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.show()


def plot_compare_multiple_zeta(t_eval, zeta_list, numerical_undamped_duffing_list, NN_TL_solution):

    title_fs  = 25
    label_fs  = 25
    tick_fs   = 18
    legend_fs = 25

    n = len(zeta_list)
    cmap = cm.get_cmap("viridis", n)

    # Two-row layout: top row = plots, bottom row = big legend
    fig = plt.figure(figsize=(5*n, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[14, 3])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]

    for i, ax in enumerate(axes):
        # Pick two contrasting colors from viridis for this subplot
        base_color = cmap(i / (n-1 if n > 1 else 1))
        rk_color   = base_color
        pinn_color = cm.viridis(0.1 + 0.8*i/(n-1 if n > 1 else 1))  # slightly shifted

        y_num = numerical_undamped_duffing_list[i][0, :]
        y_nn  = NN_TL_solution[:, i, 0]

        ax.plot(t_eval, y_num,
                label="RK45 Solution",
                color=rk_color, linewidth=2.5)

        ax.plot(t_eval, y_nn,
                label="PTL-PINN Solution",
                color=pinn_color, linewidth=2.5,
                linestyle="--", marker="o", markevery=250)

        ax.set_title(f"ζ = {zeta_list[i]}", fontsize=title_fs, pad=15)
        ax.set_xlabel("t", fontsize=label_fs, labelpad=8)
        if i == 0:
            ax.set_ylabel("x(t)", fontsize=label_fs, labelpad=10)

        ax.tick_params(axis='both', labelsize=tick_fs)
        ax.grid(alpha=0.3)

        # independent y-limits per subplot (with padding)
        y_min = min(np.min(y_num), np.min(y_nn))
        y_max = max(np.max(y_num), np.max(y_nn))
        pad = 0.05 * max(1e-12, (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)

    # Legend-only axis
    handles, labels = axes[0].get_legend_handles_labels()
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")
    leg_ax.legend(
        handles, labels,
        loc="center",
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=legend_fs,
        handlelength=3.0,
        markerscale=1.6,
        columnspacing=2.5,
        borderaxespad=1.0,
    )

    plt.show()


def plot_IAE_multiple_zeta(zeta_list, t_eval, NN_TL_solution_list, numerical_undamped_duffing_list):

    fig, ax = plt.subplots(figsize=(12, 4))

    # Colormap: one color per zeta value
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(zeta_list)))

    for i, zeta in enumerate(zeta_list):
        error = abs(NN_TL_solution_list[:, i, 0] - numerical_undamped_duffing_list[i][0])
        print(f'zeta: {zeta}, mean Error: {np.mean(error):.3e}')
        
        cumulative_error = cumulative_trapezoid(error, t_eval, initial=0) 
        ax.plot(
            t_eval,
            cumulative_error,
            color=colors[i],
            linewidth=2,
            label=fr'$\zeta = {zeta}$'
        )

    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'IAE$(t)$', fontsize=14)

    # Legend on top, horizontal layout
    ax.legend(
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(zeta_list),
        fontsize=14,
    )

    # Fewer ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Formatter: show 0 as "0", others with 3 decimal places
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: "0" if abs(x) < 1e-12 else f"{x:.3f}")
    )

    plt.tight_layout()
    plt.show()

def plot_error_by_order(t_eval, PINN_solution, numerical_list, p_list, param_list = [], param_name = ""):

    for i in range(len(PINN_solution)):

        for k in range(p_list[0]):

            error = np.abs(numerical_list[i][0, :] - PINN_solution[i][k])
            print(f"{param_name}: {param_list[i]}, order: {k}, mean error absolute: {np.mean(error)}")
            cumulative_error = cumulative_trapezoid(error, t_eval, initial=0) 
            plt.plot(t_eval, cumulative_error, label=f"{param_name}: {param_list[i]}, order: {k}")

        plt.legend()
        plt.show()


def plot_KG_solution(sol, c, t_eval, x_span, t_span, title="", w_lpm = 1):

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