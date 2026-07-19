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
    "text.usetex": False,})

def plot_numerical_LKV(x, y, t_eval):

    plt.figure(figsize=(10, 4.5))
    plt.plot(t_eval, x, label="Prey x(t)")
    plt.plot(t_eval, y, '--', label="Predator y(t)")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.figure(figsize=(5.5, 5.0))
    plt.plot(x, y, lw=2)
    plt.scatter([1], [1], s=40, zorder=5, color="k")
    plt.text(1.02, 0.97, "(1,1)", fontsize=11)
    plt.xlabel("Prey (x)")
    plt.ylabel("Predator (y)")
    plt.title("Normalized Lotka-Volterra Phase Portrait")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

def xi_x_eta(eta, xi, N):

    result = np.zeros_like(xi[0])
    for i in range(N):
        result += xi[i] * eta[N - 1 - i]

    return result

def dtau_xi_w(xi_dot, w_list, N):

    result = np.zeros_like(xi_dot[0])
    for i in range(1, N):
        result += w_list[i] * xi_dot[N - i]

    return result

def calc_B(eta, xi, xi_dot, w_list, N):
    
    return dtau_xi_w(xi_dot, w_list, N) + xi_x_eta(eta, xi, N)

def calculate_forcing_xi(w_n, w_list, eta, xi, xi_dot):

    B_term = calc_B(eta, xi, xi_dot, w_list, N=len(xi))
    w_n_term = w_n * xi_dot[0]

    return - w_n_term - B_term

def calculate_forcing_eta(w_n, w_list, eta, xi, eta_dot, alpha):

    N = len(eta)
    B_term = dtau_xi_w(eta_dot, w_list, N) + alpha * xi_x_eta(eta, xi, N)
    w_n_term = w_n * eta_dot[0]

    return - w_n_term + B_term

def calc_w_n(w_list, xi, xi_dot, eta, t_eval):

    lib = np if type(xi[0]) == np.ndarray else torch

    mask    = (t_eval >= 0) & (t_eval <= 2*lib.pi)
    t_seg   = t_eval[mask]    

    xi_seg = [term[mask] for term in xi]
    eta_seg = [term[mask] for term in eta] 
    xi_dot_seg = [term[mask] for term in xi_dot] 

    N = len(xi)
    B = calc_B(eta_seg, xi_seg, xi_dot_seg, w_list, N)
    K = - xi_dot_seg[0]

    num = lib.trapezoid(B * (eta_seg[0]), x=t_seg)
    den = lib.trapezoid(K * (eta_seg[0]), x=t_seg)

    w_n = num / den

    return w_n
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

def plot_lv_comparison(t_eval_lpm, t_eval_num, NN_TL_solution_LPM, x, y, epsilon_list):
    # --- Style settings ---
    label_fs  = 18
    tick_fs   = 14
    legend_fs = 14
    cmap = cm.get_cmap("viridis", 3)

    fig, ax = plt.subplots(figsize=(12, 4.8))

    # --- Define consistent colors for (xi,x) and (eta,y) ---
    color_xi  = cm.viridis(0.15)  # i = 0
    color_eta = cm.viridis(0.55)  # i = 1

    # --- Numerical reference first (lighter colors) ---
    ax.plot(
        t_eval_num,
        x,
        label="$x$",
        color=color_xi,
        linewidth=2.0,
        alpha=0.7,
        zorder=1,
    )

    ax.plot(
        t_eval_num,
        y,
        label="$y$",
        color=color_eta,
        linewidth=2.0,
        alpha=0.7,
        zorder=1,
    )

    # --- PTL-PINN approximations last (bolder dashed lines) ---
    ax.plot(
        t_eval_lpm,
        1 + epsilon_list[0] * NN_TL_solution_LPM[:, 0],
        label=r"$1 + \varepsilon \xi$",
        color=color_xi,
        linewidth=2.4,
        linestyle=(0, (6, 3)),  # longer dashes
        marker="o",
        markersize=4.5,
        markevery=100,
        alpha=0.95,
        zorder=3,
    )

    ax.plot(
        t_eval_lpm,
        1 + epsilon_list[0] * NN_TL_solution_LPM[:, 1],
        label=r"$1 + \varepsilon \eta$",
        color=color_eta,
        linewidth=2.4,
        linestyle=(0, (6, 3)),  
        marker="s",
        markersize=4.5,
        markevery=100,
        alpha=0.95,
        zorder=3,
    )

    # --- Axis labels and ticks ---
    ax.set_xlabel("t", fontsize=label_fs, labelpad=6)
    ax.set_ylabel("Solution", fontsize=label_fs, labelpad=8)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # --- Legend on top ---
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        fontsize=legend_fs,
        frameon=False,
        handlelength=2.8,
        markerscale=1.2,
        labelspacing=0.4,
    )

    plt.tight_layout()
    plt.show()

def calculate_error(w_sol, perturbation_solution_LPM, epsilon, alpha_list, t_span, N, ode, x0, y0, plot=True, refine=200):
    """
    Mean absolute error of the cumulative PTL-PINN prey trajectory against a
    high-accuracy numerical reference, computed per perturbation order.

    Parameters
    ----------
    plot : bool, default True
        If True, show per-order comparison plots (kept for backward compatibility).
    refine : int
        Sub-sampling factor used for the numerical reference grid.

    Returns
    -------
    error : list[float]
        MAE for orders 0, 1, ..., p (length == len(w_sol[0])).
    """
    w_final = []
    xi_final = []
    error = []

    for i in range(len(w_sol[0])):

        if i == 0:
            w_loop = np.sqrt(alpha_list[0])
            xi_loop = perturbation_solution_LPM[0][:, 0]
        else:
            w_loop = w_final[i - 1] + (epsilon ** i) * w_sol[0][i]
            xi_loop = xi_final[i - 1] + (epsilon ** i) * perturbation_solution_LPM[i][:, 0]

        w_final.append(w_loop)
        xi_final.append(xi_loop)

        t_span_loop = (0, t_span[-1] / w_loop)
        t_eval_loop_ptl = np.linspace(t_span_loop[0], t_span_loop[1], N)
        t_eval_loop = np.linspace(t_span_loop[0], t_span_loop[1], (t_eval_loop_ptl.size - 1) * refine + 1)
        sol_loop = numerical.solve_ode_equation(ode, t_span_loop, t_eval_loop, [x0, y0])
        x_loop, _ = sol_loop

        error.append(np.mean(np.abs(1 + epsilon * xi_loop - x_loop[::refine])))

        if plot:
            plt.plot(t_eval_loop, x_loop)
            plt.plot(t_eval_loop_ptl, 1 + epsilon * xi_loop)
            plt.show()

    return error


def compute_error_per_order(H_dict, training_log, epsilon, alpha, ic, p_max,
                            ode, x0, y0, t_eval, t_span, N, refine=200):
    """
    End-to-end pipeline: given a basis ``H_dict`` (pretrained PINN, randomly
    initialised network, Fourier features, ...), compute the Lindstedt-Poincaré
    perturbation solution up to order ``p_max`` and return the MAE of the
    cumulative prey trajectory at every order.

    The function is the building block used to compare basis/hyperparameter
    choices on the same Lotka-Volterra problem.

    Returns
    -------
    error : list[float]
        MAE for orders 0, 1, ..., p_max.
    w_sol : list[list[float]]
        The frequency-correction series produced by the LPM solve.
    """
    from ptlpinns.models import transfer  # local import avoids circular deps

    w_sol = []
    _, perturbation_solution, _ = transfer.compute_perturbation_solution_LKV(
        beta_list=[epsilon],
        p_list=[p_max],
        ic_list=[ic],
        alpha_list=[alpha],
        H_dict=H_dict,
        t_eval=t_eval,
        training_log=training_log,
        all_p=False,
        comp_time=False,
        w_sol=w_sol,
    )

    error = calculate_error(
        w_sol, perturbation_solution, epsilon, [alpha],
        t_span, N, ode, x0, y0, plot=False, refine=refine,
    )
    return error, w_sol


def plot_mae_vs_order(errors_by_label, title=None, save_path=None,
                      figsize=(15, 9), color_overrides=None, grid=True):
    """
    Comparison plot of Mean Absolute Error vs perturbation order for several
    basis / hyperparameter configurations.

    Parameters
    ----------
    errors_by_label : dict[str, list[float]]
        Mapping from configuration label to list of MAE values (one entry per
        perturbation order, starting at order 0).
    title : str, optional
        Optional figure title.
    save_path : str, optional
        If given, save the figure to this path before showing it.
    figsize : tuple, default (15, 9)
        Matplotlib figure size in inches.
    color_overrides : dict[str, color], optional
        Mapping from label to an explicit matplotlib colour, used to force the
        colour of specific curves (e.g. a red baseline) instead of the
        automatic viridis assignment.
    grid : bool, default True
        Whether to draw the background grid lines.
    """
    label_fs = 18
    tick_fs = 14
    legend_fs = 12
    color_overrides = color_overrides or {}

    cmap = mpl.cm.get_cmap("viridis")
    n = max(len(errors_by_label), 2)
    colors = [cmap(0.05 + 0.9 * i / (n - 1)) for i in range(n)]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, errs) in enumerate(errors_by_label.items()):
        orders = np.arange(len(errs))
        ax.semilogy(
            orders,
            errs,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=2.4,
            markersize=9,
            color=color_overrides.get(label, colors[i]),
            label=label,
        )

    max_p = max(len(v) for v in errors_by_label.values())
    ax.set_xticks(np.arange(max_p))
    ax.set_xlabel(r"Perturbation order $p$", fontsize=label_fs)
    ax.set_ylabel("Mean absolute error", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.grid(grid, which="both", alpha=0.3)
    ax.legend(
        frameon=False, fontsize=legend_fs,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        handlelength=2.6, labelspacing=0.6,
    )
    if title is not None:
        ax.set_title(title, fontsize=label_fs)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax