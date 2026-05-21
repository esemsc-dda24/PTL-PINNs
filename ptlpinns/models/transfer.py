import torch
from neurodiffeq import diff
from neurodiffeq.generators import Generator1D
import numpy as np
from matplotlib import pyplot as plt
from ptlpinns.perturbation import LPM, standard, LKV
from ptlpinns.models import training
import time
from typing import List

def compute_H_dict(model, N, bias, t_span):

    model.to('cpu')
    # define the interior set used to do transfer learning on cpu
    t = generate_eval_tensor(N=N, t_span=t_span, require_grad=True)
    # define the boundary set used to do transfer learing on cpu
    t_initial = generate_eval_tensor(N=1, t_span=(0, 0), require_grad=True)

    # compute the hidden space H evaluated at the interior
    _, H = model(t)  # shape (IG0*IG1, 2W)
    _, H_ic = model(t_initial)  # shape (Nic, 2W)
    # compute the gradients of the hidden H in the interior data points
    Ht = compute_Ht(H, t)  # of shape (IG0*IG1, 2W)
    Ht_ic = compute_Ht(H_ic, t_initial)  # shape (Nic, 2W)
    # detach the H
    H = H.detach().numpy()  # shape (N, 2W)
    H_ic = H_ic.detach().numpy()  # shape (1, 2W)
    H = H.reshape(2 * H.shape[0], -1)  # transform all H into shape (2N, W)
    H_ic = H_ic.reshape(2 * H_ic.shape[0], -1)  # shape (2, W)
    # reshape differentiation of H
    Ht = Ht.reshape(2 * Ht.shape[0], -1)  # shape (2N, W)
    Ht_ic = Ht_ic.reshape(2 * Ht_ic.shape[0], -1)  # shape (2, W)

    # now add another dimension to the H and Hx Ht if needed
    # all zeros for Hx and Ht and all ones for H
    if bias:
        H = np.hstack((H, np.ones((H.shape[0], 1))))
        H_ic = np.hstack((H_ic, np.ones((H_ic.shape[0], 1))))
        Ht = np.hstack((Ht, np.zeros((Ht.shape[0], 1))))
        Ht_ic = np.hstack((Ht_ic, np.zeros((Ht_ic.shape[0], 1))))

    # define the matrices A
    B = np.array([[1, 0], [0, 1]])
    BHt = compute_AH(B, Ht)  # shape (2N, W)

    # ------------------------------------------------------------------ #
    # Precompute constant Gram matrices so that compute_TL / compute_TL_LKV
    # can assemble  H_star.T @ H_star  via O(W²) scalar-weighted sums
    # instead of an O(N·W²) matmul on every iteration.
    #
    # H_star = BHt + AH,  where
    #   AH_even = a00*H_even + a01*H_odd
    #   AH_odd  = a10*H_even + a11*H_odd
    #
    # H_star.T @ H_star expands to:
    #   G_BtBt
    #   + a00*(G_Be_e + G_Be_e.T) + a01*(G_Be_o + G_Be_o.T)
    #   + a10*(G_Bo_e + G_Bo_e.T) + a11*(G_Bo_o + G_Bo_o.T)
    #   + (a00²+a10²)*G_ee
    #   + (a00*a01+a10*a11)*(G_eo + G_eo.T)
    #   + (a01²+a11²)*G_oo
    # ------------------------------------------------------------------ #
    H_even   = H[0::2, :]      # (N, W)
    H_odd    = H[1::2, :]      # (N, W)
    BHt_even = BHt[0::2, :]   # (N, W)
    BHt_odd  = BHt[1::2, :]   # (N, W)

    G_BtBt     = BHt.T @ BHt                              # W×W, constant
    G_ee       = H_even.T @ H_even                        # W×W, constant
    G_oo       = H_odd.T @ H_odd                          # W×W, constant
    G_sym_eo   = H_even.T @ H_odd + H_odd.T @ H_even      # W×W, symmetric
    G_sym_Be_e = BHt_even.T @ H_even + H_even.T @ BHt_even  # W×W, symmetric
    G_sym_Be_o = BHt_even.T @ H_odd  + H_odd.T @ BHt_even   # W×W, symmetric
    G_sym_Bo_e = BHt_odd.T @ H_even  + H_even.T @ BHt_odd   # W×W, symmetric
    G_sym_Bo_o = BHt_odd.T @ H_odd   + H_odd.T @ BHt_odd    # W×W, symmetric
    G_ic       = H_ic.T @ H_ic                             # W×W, constant

    H_dict = {
        'H': H, 'H_ic': H_ic, 'Ht': Ht, 'Ht_ic': Ht_ic, 'N': N, 'BHt': BHt,
        # precomputed Gram matrices
        'G_BtBt': G_BtBt,
        'G_ee': G_ee, 'G_oo': G_oo, 'G_sym_eo': G_sym_eo,
        'G_sym_Be_e': G_sym_Be_e, 'G_sym_Be_o': G_sym_Be_o,
        'G_sym_Bo_e': G_sym_Bo_e, 'G_sym_Bo_o': G_sym_Bo_o,
        'G_ic': G_ic,
    }
    return H_dict

def compute_H_dict_fourier(N, bias, t_span, n_modes=64, orthogonalize=True):
    """
    Build H_dict from a fixed Fourier feature basis — no neural network required.

    Features are sin(2π k t / T) and cos(2π k t / T) for k = 1..n_modes,
    evaluated at N equally-spaced points.  Their time derivatives are computed
    analytically, so Ht contains no discretisation error.

    When ``orthogonalize`` is True (default), the 2*n_modes raw features are
    QR-orthogonalised using the interior grid.  Because Q has orthonormal
    columns, H_even^T H_even = I and H_odd^T H_odd = I (before the optional
    bias column), giving G_ee = G_oo = I — the best possible conditioning for
    the Gram solve.  The same rotation R^{-1} is applied consistently to dPhi,
    Phi_ic and dPhi_ic so the entire H_dict is self-consistent.

    When ``orthogonalize`` is False, the raw (un-orthogonalised) sin/cos
    features are used directly.  This is closer to "classical" Fourier
    transfer learning where no preconditioning of the basis is performed, and
    is useful as an ablation against the QR-orthogonalised variant.

    Feature assignment mirrors compute_H_dict:
        even rows of H  (component 0)  <-  first  n_modes features
        odd  rows of H  (component 1)  <-  second n_modes features

    Intended as a no-pretraining baseline: drop-in replacement for the H_dict
    produced by compute_H_dict, compatible with compute_TL_LKV and
    compute_perturbation_solution_LKV.

    Parameters
    ----------
    N             : number of interior collocation points
    bias          : if True, append a bias column (ones to H/H_ic, zeros to Ht/Ht_ic)
    t_span        : (t0, T)
    n_modes       : number of Fourier frequencies; when ``orthogonalize`` is
                    True it is clipped to N//2 so the QR is full-rank.
    orthogonalize : if True, QR-orthogonalise the basis; if False, use raw
                    sin/cos features.
    """
    if orthogonalize:
        n_modes = min(n_modes, N // 2)

    T  = t_span[1] - t_span[0]
    t0 = t_span[0]
    omegas = 2.0 * np.pi * np.arange(1, n_modes + 1) / T

    def _raw_features(t_grid):
        """Raw sin/cos features and their exact time derivatives."""
        tau  = (t_grid - t0)[:, None] * omegas[None, :]          # (M, n_modes)
        Phi  = np.concatenate([np.sin(tau), np.cos(tau)],   1)   # (M, 2*n_modes)
        dPhi = np.concatenate([omegas * np.cos(tau),
                               -omegas * np.sin(tau)],       1)
        return Phi, dPhi

    t_vals = np.linspace(t_span[0], t_span[1], N)
    t_ic   = np.array([t_span[0]])

    Phi,    dPhi    = _raw_features(t_vals)   # (N, 2*n_modes)
    Phi_ic, dPhi_ic = _raw_features(t_ic)     # (1, 2*n_modes)

    if orthogonalize:
        # QR on interior: Phi = Q R  =>  Q = Phi @ R^{-1}
        # N >= 2*n_modes is guaranteed, so R is square and invertible.
        Q, R  = np.linalg.qr(Phi, mode='reduced')        # Q: (N, 2k), R: (2k, 2k)
        R_inv = np.linalg.solve(R, np.eye(R.shape[0]))

        Phi_orth     = Q                     # (N, 2*n_modes), orthonormal columns
        dPhi_orth    = dPhi    @ R_inv       # (N, 2*n_modes)
        Phi_ic_orth  = Phi_ic  @ R_inv       # (1, 2*n_modes)
        dPhi_ic_orth = dPhi_ic @ R_inv       # (1, 2*n_modes)
    else:
        # Raw (un-orthogonalised) sin/cos features.
        Phi_orth     = Phi
        dPhi_orth    = dPhi
        Phi_ic_orth  = Phi_ic
        dPhi_ic_orth = dPhi_ic

    # Split features between the two ODE components and interleave to match
    # the (2N, W) layout produced by compute_H_dict.
    #   row 2i   <- first  n_modes features at t_i  (component 0)
    #   row 2i+1 <- second n_modes features at t_i  (component 1)
    W = n_modes   # features per component
    H     = Phi_orth[:, :2*W].reshape(2 * N, W)
    Ht    = dPhi_orth[:, :2*W].reshape(2 * N, W)
    H_ic  = Phi_ic_orth[:, :2*W].reshape(2, W)
    Ht_ic = dPhi_ic_orth[:, :2*W].reshape(2, W)

    if bias:
        H     = np.hstack((H,    np.ones ((H.shape[0],    1))))
        H_ic  = np.hstack((H_ic, np.ones ((H_ic.shape[0], 1))))
        Ht    = np.hstack((Ht,   np.zeros((Ht.shape[0],   1))))
        Ht_ic = np.hstack((Ht_ic, np.zeros((Ht_ic.shape[0], 1))))

    B   = np.array([[1, 0], [0, 1]])
    BHt = compute_AH(B, Ht)

    H_even   = H[0::2, :]
    H_odd    = H[1::2, :]
    BHt_even = BHt[0::2, :]
    BHt_odd  = BHt[1::2, :]

    return {
        'H': H, 'H_ic': H_ic, 'Ht': Ht, 'Ht_ic': Ht_ic, 'N': N, 'BHt': BHt,
        'G_BtBt':     BHt.T @ BHt,
        'G_ee':       H_even.T @ H_even,
        'G_oo':       H_odd.T @ H_odd,
        'G_sym_eo':   H_even.T @ H_odd + H_odd.T @ H_even,
        'G_sym_Be_e': BHt_even.T @ H_even + H_even.T @ BHt_even,
        'G_sym_Be_o': BHt_even.T @ H_odd  + H_odd.T @ BHt_even,
        'G_sym_Bo_e': BHt_odd.T @ H_even  + H_even.T @ BHt_odd,
        'G_sym_Bo_o': BHt_odd.T @ H_odd   + H_odd.T @ BHt_odd,
        'G_ic':       H_ic.T @ H_ic,
    }


def compute_Ht(H, t):
    output = []
    for i in range(H.shape[1]):
        output.append(diff(H[:, i].reshape(-1, 1), t).detach().numpy())
    return np.concatenate(output, axis=1)

def compute_AH(A, H):
    N, W_size = H.shape
    A_reshaped = A.reshape(1, 2, 2)
    # Reshape H to (3600, 2, 256) to match the dimensions of A
    H_reshaped = H.reshape(-1, 2, W_size)
    # Perform the multiplication
    AH = np.matmul(A_reshaped, H_reshaped)
    # Reshape the result back
    AH = AH.reshape(-1, W_size)
    return AH

# this function generate eval tensors to evaluate the Hs
def generate_eval_tensor(N=512, t_span=(0, 1), require_grad=True):
    generator = Generator1D(size=N, method="equally-spaced", t_min=t_span[0], t_max=t_span[1])
    t = generator.get_examples().unsqueeze(1)  # (N, 1)
    # convert this sample points into input to the network and requires gradients
    t = t.cpu()
    if require_grad:
        t.requires_grad_()
    return t

def _gram_H_star(A, H_dict):
    """Return H_star.T @ H_star using precomputed W×W Gram matrices.

    Expands (BHt + AH).T (BHt + AH) with
        AH_even = a00*H_even + a01*H_odd
        AH_odd  = a10*H_even + a11*H_odd
    giving an O(W²) operation instead of the O(N·W²) explicit matmul.
    """
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]
    return (
        H_dict['G_BtBt']
        + a00 * H_dict['G_sym_Be_e']
        + a01 * H_dict['G_sym_Be_o']
        + a10 * H_dict['G_sym_Bo_e']
        + a11 * H_dict['G_sym_Bo_o']
        + (a00 ** 2 + a10 ** 2) * H_dict['G_ee']
        + (a00 * a01 + a10 * a11) * H_dict['G_sym_eo']
        + (a01 ** 2 + a11 ** 2) * H_dict['G_oo']
    )


def compute_perturbation_solution(w_0_list, zeta_list, beta_list, p_list, ic_list, forcing_list, H_dict, t_eval, training_log, all_p=False, comp_time=False, solver="LPM", w_sol = [], power=[(3, 1)], invert=True):

    NN_TL_solution = []
    TL_comp_time = []
    perturbation_solution_list = []
    for i, (w_0_transfer, zeta_transfer, beta_transfer) in enumerate(zip(w_0_list, zeta_list, beta_list)):
        NN_TL_solution_w_0 = []
        for p in p_list if all_p else [p_list[i]]:
            perturbation_solution = []

            if solver == "LPM":
                x_ddot, x_lin = [], []
                if w_sol == []:
                    w_sol.append([1])

            for j in range(p+1):
                if j==0:
                    W, TL_time = compute_TL(w_0=w_0_transfer, zeta=zeta_transfer, forcing_function=forcing_list[i], ic=ic_list[i],
                                                        w_ode=training_log['w_ode'], w_ic=training_log['w_ic'], H_dict=H_dict, t=t_eval, invert=invert)
                    H_dict["R_ic"] = np.zeros_like(H_dict["R_ic"])
                    perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
                else:
                    if solver == "standard":

                        forcing_time = time.perf_counter()
                        #force_function_index = standard.force_func_perturbation(j)
                        #force_perturbation = 0
                        force_perturbation = standard.calculate_forcing(j, power, perturbation_solution)
                        #for (a, b, c, coefficient) in force_function_index:
                        #    force_perturbation -= coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0]*perturbation_solution[c][:, 0]
                        
                        force_perturbation = np.stack((np.zeros_like(force_perturbation), force_perturbation), axis=1)


                        TL_time += time.perf_counter() - forcing_time

                    elif solver == "LPM":
                        if type(w_sol) == np.ndarray or type(w_sol) == list:

                            forcing_time = time.perf_counter()
                            x_lin.append(perturbation_solution[-1][:, 0])
                            x_ddot.append(np.gradient(perturbation_solution[-1][:, 1], t_eval))

                            if len(w_sol[i]) <= j:
                                w_n = LPM.calc_w_n(w_list=w_sol[i], x=x_lin, x_ddot=x_ddot, t=t_eval, power=power)
                                w_sol[i].append(w_n)
                            else:
                                w_n = w_sol[i][j]

                            x_n_forcing = LPM.calculate_forcing(w_n=w_n, w_list=w_sol[i], x=x_lin, x_ddot=x_ddot, power=power)
                            force_perturbation = np.stack((np.zeros_like(x_n_forcing), x_n_forcing), axis=1)
                            TL_time += time.perf_counter() - forcing_time
                        else:
                            raise ValueError("w_sol should either be provided as a list or numpy array")

                    compute_start = time.perf_counter()
                    W = compute_TL_with_F(force_perturbation, w_ode=training_log['w_ode'], H_dict=H_dict)
                    perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
                    compute_time = time.perf_counter() - compute_start
                    TL_time += compute_time

            perturbation_solution_list.append(perturbation_solution)
            NN_TL_solution_w_0.append(sum([(beta_transfer**k)*perturbation_solution[k] for k in range(p+1)]))
            TL_comp_time.append(TL_time)

        NN_TL_solution.append(np.stack(NN_TL_solution_w_0, axis=0).squeeze())
    NN_TL_solution = np.stack(NN_TL_solution, axis=1 if all_p else 0)
    if comp_time:
        return NN_TL_solution, H_dict, TL_comp_time
    else:
        if len(zeta_list) == 1:
            return NN_TL_solution, perturbation_solution, H_dict
        else:
            return NN_TL_solution, perturbation_solution_list, H_dict
    

def compute_TL(w_0, zeta, ic, forcing_function, w_ode, w_ic, H_dict, t=None, invert=True):
    A = get_A(w_0=w_0, zeta=zeta)
    AH = compute_AH(A, H_dict['H'])
    H_star = H_dict["BHt"] + AH
    H_dict["H_star"] = H_star
    N = H_dict['N']
    start_time = time.perf_counter()

    if invert:
        M = w_ode * _gram_H_star(A, H_dict) / N + w_ic * H_dict['G_ic']  # shape (W, W)
        Minv = np.linalg.pinv(M)
        H_dict["M_inv"] = Minv

    # forcing function
    if t is not None:
        forcing_value = forcing_function(t).reshape(-1, 1)
        Rf = w_ode * (H_star.T @ forcing_value) / N
    else:
        Rf = w_ode * (H_star.T @ forcing_function) / N
    H_dict["Rf"] = Rf

    # initial condition
    Ric = w_ic * ((ic * H_dict['H_ic'].T).sum(axis=1)).reshape(-1, 1)
    H_dict["R_ic"] = Ric

    # compute W
    R = Rf + Ric
    W = H_dict["M_inv"] @ R  # shape (256, 1)
    computational_time = time.perf_counter() - start_time

    return W, computational_time


def compute_TL_with_F(forcing_function, w_ode, H_dict, t=None):
    # forcing function
    if t is not None:
        forcing_value = forcing_function(t).reshape(-1, 1)
        Rf = w_ode * (H_dict["H_star"].T @ forcing_value) / H_dict['N']
    else:
        Rf = w_ode * (H_dict["H_star"].T @ forcing_function.reshape(-1, 1)) / H_dict["N"]
    
    # compute W
    R = Rf + H_dict['R_ic']
    W = H_dict['M_inv'] @ R  # shape (256, 1)

    return W

def get_A(w_0, zeta=0):
    return np.array([[0, -1], [w_0 ** 2, 2 * zeta * w_0]])

def compute_solution(H, W, N):
    return (H @ W).reshape(N, -1).T


def evaluate_MAE(transfer_model, numerical_solution, t_span, N):
    """
    Evaluate the Mean Absolute Error (MAE) between the model's prediction and the numerical solution.
    Stopping condition for transfer learning.
    """

    t_eval_torch = training.generate_eval_tensor(N, t_span, False)
    model_result = transfer_model(t_eval_torch)[0].squeeze()[:, 0].detach().numpy()
    return np.mean(np.abs(model_result - numerical_solution[0, :]))


def compute_transfer_learning(transfer_model, optimizer, num_iter, equation_functions, 
                            initial_condition_functions, forcing_functions,
                            numerical_solution, N=512, t_span=(0, 1),
                            every=100, ode_weight=1, ic_weight=1,
                            method='equally-spaced-noisy', scheduler=None):
       """
       Regular Multi-Headed-PINN transfer learning.
       """

       total_time = 0.0

       for it in range(1, num_iter):

              start_time = time.perf_counter()
              optimizer.zero_grad()
              total, ode, ic, _ = training.loss(
              model=transfer_model, N=N, t_span=t_span,
              equation_functions=equation_functions,
              initial_condition_functions=initial_condition_functions,
              forcing_functions=forcing_functions,
              ode_weight=ode_weight, ic_weight=ic_weight, method=method)
              total.backward()
              optimizer.step()

              if scheduler is not None:
                     scheduler.step()

              end_time = time.perf_counter()
              total_time += end_time - start_time

              MAE = evaluate_MAE(transfer_model, numerical_solution, t_span, N)

              if MAE < 2.5e-2:
                     print(f"Converged at iteration {it}: MAE = {MAE} | time {total_time}")
                     break

              if it % every == 0:
                     print(f"[iteration] {it} | total {total.item():.3e} | ode {ode.item():.3e} | ic {ic.item():.3e} | MAE {MAE:.3e} | time {total_time:.2f}")

def get_A_LKV(alpha):
    return np.array([[0, 1/np.sqrt(alpha)], [-np.sqrt(alpha), 0]])

def compute_TL_LKV(alpha, ic, w_ode, w_ic, H_dict, invert=True):
    A = get_A_LKV(alpha)
    AH = compute_AH(A, H_dict['H'])
    H_star = H_dict["BHt"] + AH
    H_dict["H_star"] = H_star
    N = H_dict['N']
    start_time = time.perf_counter()

    if invert:
        M = w_ode * _gram_H_star(A, H_dict) / N + w_ic * H_dict['G_ic']  # shape (W, W)
        Minv = np.linalg.pinv(M)
        H_dict["M_inv"] = Minv

    H_dict["Rf"] = 0

    # initial condition
    Ric = w_ic * ((ic * H_dict['H_ic'].T).sum(axis=1)).reshape(-1, 1)
    H_dict["R_ic"] = Ric

    # compute W
    R = Ric
    W = H_dict["M_inv"] @ R  # shape (256, 1)
    computational_time = time.perf_counter() - start_time

    return W, computational_time


def compute_perturbation_solution_LKV(beta_list, p_list, ic_list, alpha_list, H_dict, t_eval, training_log, all_p=False, comp_time=False, w_sol = [], invert=True):

    NN_TL_solution = []
    TL_comp_time = []
    perturbation_solution_list = []
    for i, (alpha_transfer, beta_transfer) in enumerate(zip(alpha_list, beta_list)):
        NN_TL_solution_w_0 = []
        for p in p_list if all_p else [p_list[i]]:
            perturbation_solution = []

            xi_list, eta_list, xi_dot, eta_dot = [], [], [], []
            if w_sol == []:
                w_sol.append([np.sqrt(alpha_transfer)])

            for j in range(p+1):
                if j==0:

                    W, TL_time = compute_TL_LKV(alpha = alpha_transfer, ic=ic_list[i], 
                                                w_ode=training_log['w_ode'], w_ic=training_log['w_ic'],
                                                H_dict=H_dict, invert=invert)
                    
                    H_dict["R_ic"] = np.zeros_like(H_dict["R_ic"])
                    perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
                else:

                    if type(w_sol) == np.ndarray or type(w_sol) == list:

                        forcing_time = time.perf_counter()

                        xi_list.append(perturbation_solution[-1][:, 0])
                        xi_dot.append(np.gradient(perturbation_solution[-1][:, 0], t_eval))
                        eta_list.append(perturbation_solution[-1][:, 1])
                        eta_dot.append(np.gradient(perturbation_solution[-1][:, 1], t_eval))

                        if len(w_sol[i]) <= j:
                            w_n = LKV.calc_w_n(w_sol[i], xi_list, xi_dot, eta_list, t_eval)
                            w_sol[i].append(w_n)
                        else:
                            w_n = w_sol[i][j]

                        # print(w_n)

                        xi_forcing = LKV.calculate_forcing_xi(w_n, w_sol[i], eta_list, xi_list, xi_dot)
                        eta_forcing = LKV.calculate_forcing_eta(w_n, w_sol[i], eta_list, xi_list, eta_dot, alpha_transfer)
                        
                        #plt.plot(t_eval, xi_forcing, label=f"xi forcing order {j}")
                        #plt.legend()
                        #plt.show()

                        #plt.plot(t_eval, eta_forcing, label=f"eta forcing order {j}")
                        #plt.legend()
                        #plt.show()

                        force_perturbation = np.stack((xi_forcing / np.sqrt(alpha_transfer), eta_forcing / np.sqrt(alpha_transfer)), axis=1)
                        TL_time += time.perf_counter() - forcing_time
                    else:
                        raise ValueError("w_sol should either be provided as a list or numpy array")

                    compute_start = time.perf_counter()
                    W = compute_TL_with_F(force_perturbation, w_ode=training_log['w_ode'], H_dict=H_dict)
                    perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)

                    #plt.plot(t_eval, perturbation_solution[-1][:, 0], label=f"xi order {j}")
                    #plt.legend()
                    #plt.show()

                    #plt.plot(t_eval, perturbation_solution[-1][:, 1], label=f"eta order {j}")
                    #plt.legend()
                    #plt.show()

                    compute_time = time.perf_counter() - compute_start
                    TL_time += compute_time

            perturbation_solution_list.append(perturbation_solution)
            NN_TL_solution_w_0.append(sum([(beta_transfer**k)*perturbation_solution[k] for k in range(p+1)]))
            TL_comp_time.append(TL_time)

        NN_TL_solution.append(np.stack(NN_TL_solution_w_0, axis=0).squeeze())
    NN_TL_solution = np.stack(NN_TL_solution, axis=1 if all_p else 0)
    if comp_time:
        return NN_TL_solution, H_dict, TL_comp_time
    else:
        if len(alpha_list) == 1:
            return NN_TL_solution, perturbation_solution, H_dict
        else:
            return NN_TL_solution, perturbation_solution_list, H_dict