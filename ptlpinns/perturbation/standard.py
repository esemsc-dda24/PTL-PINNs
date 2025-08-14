import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.integrate import cumulative_trapezoid
from matplotlib.ticker import MultipleLocator


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