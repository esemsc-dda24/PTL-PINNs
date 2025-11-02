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