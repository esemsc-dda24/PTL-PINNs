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
                           perturbation_solution, t_eval, selected_orders=[0, 1, 6]):

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

    plt.ylim(0, 25)
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