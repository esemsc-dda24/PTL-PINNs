import numpy as np
import torch
from typing import List

def forcing_coeff_random(n_forcing: int, mean: float = 1):
    """
    Generate random coefficients for forcing functions.
    
    Parameters
    ----------
    n_forcing : int
        Number of forcing coefficients to generate.
    mean : float, default=1
        Expected mean value of the sum of all coefficients.
    
    Returns
    -------
    list of float
        List of random coefficients sampled from uniform distribution.
    
    Notes
    -----
    Coefficients are sampled from a uniform distribution U(0, 2*mean/n_forcing).
    This ensures that for a sufficiently large number of coefficients, the expected
    sum converges to the specified mean value:
    """

    np.random.seed(42)
    coeff = []
    for _ in range(n_forcing):
        coeff.append(np.random.uniform(0, (mean * 2)/n_forcing))
    return coeff


def no_forcing(numpy=False):
    """
    Returns 2D zero forcing function for numpy or torch tensors
    """
    def force(t):
        if numpy:
            if np.isscalar(t):
                return np.array([0.0, 0.0])
            else:
                return np.stack((np.zeros_like(t), np.zeros_like(t)), axis=1)
        else:
            if torch.is_tensor(t) and t.dim() == 0:
                return torch.tensor([0.0, 0.0])
            else:
                return torch.stack((torch.zeros_like(t), torch.zeros_like(t)), dim=1)
    return force

def sum_cosine_1D(t, numpy, w_0, coeff):
    """
    Compute the sum of cosine functions at time t
    """

    lib = np if numpy else torch
    result = lib.zeros_like(t)

    for w, A in zip(w_0, coeff):
        result += A * lib.cos(w * t)

    return result

def underdamped_1D(t, numpy: bool, w_0: List, coeff: List, mu: float, zeta: float):
    """
    Compute the underdamped forcing function, considering multiple cosine terms
    """

    lib = np if numpy else torch
    result = lib.zeros_like(t)

    for k in range(len(w_0)):
        result += coeff[k] * lib.cos(w_0[k] * lib.sqrt(1 - zeta ** 2) * t) * lib.exp(- mu * zeta * t)

    return result

def overdamped_1st_order_1D(t, numpy, zeta, w_0, coeff):
    """
    Computes the forcing function of the general 1st-order equation of the overdamped system.
    """

    lib = np if numpy else torch
    lambda_1 = - zeta * w_0 + lib.sqrt(zeta ** 2 - 1)
    lambda_2 = - zeta * w_0 - lib.sqrt(zeta ** 2 - 1)
    coeff_1 = (coeff ** 3) * lambda_2 / (lambda_2 - lambda_1)
    coeff_2 = (coeff ** 3) * lambda_1 / (lambda_1 - lambda_2)

    return coeff_1 * lib.exp(lambda_1 * t) + coeff_2 * lib.exp(lambda_2 * t)

def overdamped_1st_order_2D(numpy: bool, zeta: float, w_0: List, coeff: List):
    """
    Computes the forcing function of the general 1st-order equation of the overdamped system in 2D
    """

    def force(t):
        if numpy:
            if np.isscalar(t):
                return np.array([overdamped_1st_order_1D(t, numpy, zeta, w_0, coeff), 0.0])  # shape: (2,)
            else:
                return np.stack((overdamped_1st_order_1D(t, numpy, w_0, coeff), np.zeros_like(t)), axis=1)  # shape: (len(t), 2)
        else:
            if torch.is_tensor(t) and t.dim() == 0:
                return torch.tensor([overdamped_1st_order_1D(t, numpy, w_0, coeff), 0.0])
            else:
                return torch.stack((overdamped_1st_order_1D(t, numpy, w_0, coeff), torch.zeros_like(t)), dim=1)

    return force

def sum_cosine_forcing(numpy: bool, w_0: List, coeff: List):
    """
    Returns a 2D sum of cosine forcing function for numpy or torch tensors
    """

    def force(t):
        if numpy:
            if np.isscalar(t):
                return np.array([sum_cosine_1D(t, numpy, w_0, coeff), 0.0])  # shape: (2,)
            else:
                return np.stack((sum_cosine_1D(t, numpy, w_0, coeff), np.zeros_like(t)), axis=1)  # shape: (len(t), 2)
        else:
            if torch.is_tensor(t) and t.dim() == 0:
                return torch.tensor([sum_cosine_1D(t, numpy, w_0, coeff), 0.0])
            else:
                return torch.stack((sum_cosine_1D(t, numpy, w_0, coeff), torch.zeros_like(t)), dim=1)

    return force

def underdamped_forcing(numpy: bool, w_0: List, coeff: List, mu: float, zeta: float):
    """
    Returns a 2D general underdamped forcing function for numpy or torch tensors
    """
    def force(t):
        if numpy:
            if np.isscalar(t):
                return np.array([underdamped_1D(t, True, w_0, coeff, mu, zeta), 0.0])  # shape: (2,)
            else:
                return np.stack((underdamped_1D(t, True, w_0, coeff, mu, zeta), np.zeros_like(t)), axis=1)  # shape: (len(t), 2)
        else:
            if torch.is_tensor(t) and t.dim() == 0:
                return torch.tensor([underdamped_1D(t, False, w_0, coeff, mu, zeta), 0.0], device=t.device)
            else:
                return torch.stack((underdamped_1D(t, False, w_0, coeff, mu, zeta), torch.zeros_like(t, device=t.device)), dim=1)

    return force