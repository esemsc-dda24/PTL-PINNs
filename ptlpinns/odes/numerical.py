from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import math

def solve_ode_equation(ode, t_span, t_eval, y0, method="RK45", rtol=1e-10, atol=1e-10):
    """
    Solves an ODE definied by function `ode` using `solve_ivp`
    """
    solution = solve_ivp(ode, t_span=t_span, y0=y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    return solution.y

def plot_ode_solution(solution, t_eval, title=None, xlabel="Time", ylabel="Solution", legend=True):
    plt.figure(figsize=(6, 4))
    plt.plot(t_eval, solution[0], label="x(t)")
    plt.plot(t_eval, solution[1], label="x'(t)")
    plt.title(title if title else "ODE Solutions")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_multiple_ode_solutions(
    solutions, t_eval, titles=None, suptitle=None,
    xlabel="Time", ylabel="Amplitude", legend=True,
    ncols=2, figsize=(14, 10), hspace=0.6, wspace=0.4,
    top=0.92, bottom=0.1, left=0.08, right=0.95):
    """
    Plots multiple ODE solutions in a grid layout
    """

    n = len(solutions)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize if isinstance(figsize, tuple) else (figsize * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, sol in enumerate(solutions):
        ax = axes[i]
        ax.plot(t_eval, sol[0], label="x(t)")
        ax.plot(t_eval, sol[1], label="x'(t)")
        ax.set_title(titles[i] if titles and i < len(titles) else f"Solution {i+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if legend:
            ax.legend()

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    # Manually control subplot spacing to allow space for titles and legends
    fig.subplots_adjust(
        top=top, bottom=bottom, left=left, right=right,
        hspace=hspace, wspace=wspace
    )

    plt.show()
