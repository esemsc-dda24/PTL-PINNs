import numpy as np
import torch

def ode_oscillator_1D(w_0, zeta, forcing_1D, epsilon, q = [(3, 1)]):

    def ode(t, y):

        nonlinear = epsilon * sum(power[1] * (y[0] ** power[0]) for power in q)
        return np.array([y[1], - (w_0 ** 2) * y[0] - 2 * w_0 * zeta * y[1] - nonlinear + forcing_1D(t)])
    
    return ode

def ode_oscillator(forcing, w_0=1, epsilon=0, q = 3, zeta = 0, numpy=True):
    """
    General 2D formulation of a nonlinear oscillator (linear when epsilon = 0).

    Returns:
        function
    """

    def ode(t, y):
        force = forcing(t)
        if numpy:
            return np.array([
                y[1] + force[1],
                - (w_0 ** 2) * y[0] - epsilon * (y[0] ** q) - (2*w_0*zeta) * y[1] + force[0]
            ])
        else:
            return torch.stack([
                y[1] + force[1],
                - (w_0 ** 2) * y[0] - epsilon * (y[0] ** q) - (2*w_0*zeta) * y[1] + force[0]])
    return ode

def equation_oscillator(w_0=1.0, epsilon=0.0, zeta=0):
    def equation(y1, y2):
        return torch.stack((-y2, (w_0 ** 2) * y1 + epsilon * (y1 ** 3) + (2*w_0*zeta) * y2), dim = 1)
    return equation


def lv_normalized(alpha):
    """
    Normalized version of the Lotka-Volterra equation.

    Returns:
        function
    """

    def lv(t, z):
        x, y = z
        dx = x * (1 - y)
        dy = alpha * y * (x - 1)
        return [dx, dy]

    return lv