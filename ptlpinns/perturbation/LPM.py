import numpy as np
from matplotlib import pyplot as plt
from typing import List
import matplotlib as mpl

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
    steps = np.arange(1, len(frequency_error) + 1)

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