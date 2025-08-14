import torch
from neurodiffeq import diff
from neurodiffeq.generators import Generator1D
import numpy as np
from matplotlib import pyplot as plt
from ptlpinns.perturbation import LPM, standard
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
    H_dict = {'H': H, 'H_ic': H_ic, 'Ht': Ht, 'Ht_ic': Ht_ic, 'N': N,
              'BHt': BHt}
    return H_dict

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

def compute_perturbation_solution(w_0_list, zeta_list, beta_list, p_list, ic_list, forcing_list, H_dict, t_eval, training_log, all_p=False, comp_time=False, solver="LPM", w_sol = [], power=3):

    NN_TL_solution = []
    TL_comp_time = []
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
                    W, TL_time, H_dict_new = compute_TL(w_0=w_0_transfer, zeta=zeta_transfer, forcing_function=forcing_list[i], ic=ic_list[i],
                                                        w_ode=training_log['w_ode'], w_ic=training_log['w_ic'], H_dict=H_dict, t=t_eval)
                    H_dict_new["R_ic"] = np.zeros_like(H_dict_new["R_ic"])
                    perturbation_solution.append(compute_solution(H_dict_new['H'], W, H_dict_new['N']).T)
                else:
                    if solver == "standard":
                        force_function_index = standard.force_func_perturbation(j)
                        force_perturbation = 0
                        for (a, b, c, coefficient) in force_function_index:
                            force_perturbation -= coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0]*perturbation_solution[c][:, 0]
                        force_perturbation = np.stack((np.zeros_like(force_perturbation), force_perturbation), axis=1)

                    elif solver == "LPM":
                        if type(w_sol) == np.ndarray or type(w_sol) == list:
                            x_lin.append(perturbation_solution[-1][:, 0])
                            x_ddot.append(np.gradient(perturbation_solution[-1][:, 1], t_eval))

                            if len(w_sol[i]) <= j:
                                w_n = LPM.calc_w_n(w_list=w_sol[i], x=x_lin, x_ddot=x_ddot, t=t_eval, power=power)
                                w_sol[i].append(w_n)
                            else:
                                w_n = w_sol[i][j]

                            x_n_forcing = LPM.calculate_forcing(w_n=w_n, w_list=w_sol[i], x=x_lin, x_ddot=x_ddot, power=power)
                            force_perturbation = np.stack((np.zeros_like(x_n_forcing), x_n_forcing), axis=1)
                        else:
                            raise ValueError("w_sol should either be provided as a list or numpy array")

                    W, time = compute_TL_with_F(force_perturbation, w_ode=training_log['w_ode'], H_dict=H_dict_new)
                    perturbation_solution.append(compute_solution(H_dict_new['H'], W, H_dict_new['N']).T)
                    TL_time += time
            NN_TL_solution_w_0.append(sum([(beta_transfer**k)*perturbation_solution[k] for k in range(p+1)]))
        NN_TL_solution.append(np.stack(NN_TL_solution_w_0, axis=0).squeeze())
        TL_comp_time.append(TL_time)
    NN_TL_solution = np.stack(NN_TL_solution, axis=1 if all_p else 0)
    if comp_time:
        return NN_TL_solution, H_dict_new, TL_comp_time
    else:
        return NN_TL_solution, perturbation_solution, H_dict_new
    

def compute_TL(w_0, zeta, ic, forcing_function, w_ode, w_ic, H_dict, t=None):
    A = get_A(w_0=w_0, zeta=zeta)
    AH = compute_AH(A, H_dict['H'])
    H_star = H_dict["BHt"] + AH
    H_dict["H_star"] = H_star
    N = H_dict['N']
    H_ic_0 = H_dict['H_ic']
    start_time = time.time()

    if "M_inv" not in H_dict:
        M = w_ode * (H_star.T @ H_star) / N + w_ic * (H_ic_0.T @ H_ic_0)  # shape (W, W)
        Minv = np.linalg.pinv(M)
        H_dict["M_inv"] = Minv
    else:
        Minv = H_dict["M_inv"]

    # forcing function
    if t is not None:
        forcing_value = forcing_function(t).reshape(-1, 1)
        Rf = w_ode * (H_star.T @ forcing_value) / N
    else:
        Rf = w_ode * (H_star.T @ forcing_function) / N
    H_dict["Rf"] = Rf

    # initial condition
    Ric = w_ic * ((ic * H_ic_0.T).sum(axis=1)).reshape(-1, 1)
    H_dict["R_ic"] = Ric

    # compute W
    R = Rf + Ric
    W = Minv @ R  # shape (256, 1)
    computational_time = time.time() - start_time

    return W, computational_time, H_dict


def compute_TL_with_F(forcing_function, w_ode, H_dict, t=None):
    start_time = time.time()
    # forcing function
    if t is not None:
        forcing_value = forcing_function(t).reshape(-1, 1)
        Rf = w_ode * (H_dict["H_star"].T @ forcing_value) / H_dict['N']
    else:
        Rf = w_ode * (H_dict["H_star"].T @ forcing_function.reshape(-1, 1)) / H_dict["N"]
    
    # compute W
    R = Rf + H_dict['R_ic']
    W = H_dict['M_inv'] @ R  # shape (256, 1)
    computational_time = time.time() - start_time

    return W, computational_time

def get_A(w_0, zeta=0):
    return np.array([[0, -1], [w_0 ** 2, 2 * zeta * w_0]])

def compute_solution(H, W, N):
    return (H @ W).reshape(N, -1).T