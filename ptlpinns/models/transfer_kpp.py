from typing import List
from collections import Counter
from math import factorial
from itertools import combinations_with_replacement
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neurodiffeq import diff
from neurodiffeq.generators import Generator1D
import numpy as np
from matplotlib import pyplot as plt
import time


def generate_initial_tensor(Nic, x_span, require_grad=True, method='equally-spaced'):
    x_samples = Generator1D(size=Nic, method=method, t_min=x_span[0], t_max=x_span[1]).get_examples()
    x_initial = x_samples.unsqueeze(1)
    t_initial = torch.zeros_like(x_samples).unsqueeze(1)
    x_initial = x_initial.cpu()
    t_initial = t_initial.cpu()

    if require_grad:
        x_initial.requires_grad_()
        t_initial.requires_grad_()

    initial_tensor = torch.cat([x_initial, t_initial], dim=1)  # (N_initial, 2)
    return (x_initial, t_initial, initial_tensor)

def generate_boundary_tensor(Nbc, t_span, x, require_grad=True, method='equally-spaced'):
    ''' x is where the boundary (float) condition is applied, t_span is the time span of the problem
    '''
    t_samples = Generator1D(size=Nbc, method=method, t_min=t_span[0], t_max=t_span[1]).get_examples() # t_max is 1 because time is normalized to [0,1]
    
    t_boundary = t_samples.unsqueeze(1)
    x_boundary = x*torch.ones_like(t_samples).unsqueeze(1)
    x_boundary = x_boundary.cpu()
    t_boundary = t_boundary.cpu()

    if require_grad:
        x_boundary.requires_grad_()
        t_boundary.requires_grad_()

    boundary_tensor = torch.cat([x_boundary, t_boundary], dim=1)  # (N_initial, 2)
    return (x_boundary, t_boundary, boundary_tensor)

def generate_interior_tensor(IG=(30, 30), x_span=(0, 1),
                             t_span=(0, 1), require_grad=True):
    generator = Generator2D(grid=IG, method='equally-spaced',
                            xy_min=(x_span[0], t_span[0]),
                            xy_max=(x_span[-1], t_span[-1]))
    samples = generator.get_examples()
    # convert this sample points into input to the network and requires gradients
    x = samples[0].unsqueeze(1)  # (N, 1)
    t = samples[1].unsqueeze(1)  # (N, 1)
    x = x.cpu()
    t = t.cpu()
    if require_grad:
        x.requires_grad_()
        t.requires_grad_()
    interior_tensor = torch.cat([x, t], dim=1)
    return (x, t, interior_tensor)


# function to compute the gradient of H w.r.t. x
def compute_H2x(H, x):
    output = []
    for i in range(H.shape[1]):
        output.append(diff(H[:, i].reshape(-1, 1), x, order = 2).detach().numpy())
    return np.concatenate(output, axis=1)


# function to compute the gradient of H w.r.t. t
def compute_Ht(H, t):
    output = []
    for i in range(H.shape[1]):
        output.append(diff(H[:, i].reshape(-1, 1), t).detach().numpy())
    return np.concatenate(output, axis=1)


def compute_H_dict(model, IG, Nic,Nbc, bias, x_span, t_span, log, D):
    # change the model to cpu
    model.to('cpu')
    # generate samples
    # define the interior set used to do transfer learning on cpu
    x, t, interior_tensor = generate_interior_tensor(IG=IG, x_span=x_span, t_span=t_span, require_grad=True)

    #define the initial set used to do transfer learning on cpu
    _, _, initial_tensor = generate_initial_tensor(Nic=Nic, x_span=x_span, require_grad=True, method='equally-spaced')
    
    _, _, boundary_tensor_left = generate_boundary_tensor(Nbc=Nbc, t_span=t_span, x = 0, require_grad=True, method='equally-spaced')
    _, _, boundary_tensor_right = generate_boundary_tensor(Nbc=Nbc, t_span=t_span, x = log['domain_info']['L'], require_grad=True, method='equally-spaced')
   
   
    # compute the hidden space H evaluated at the interior
    _, H = model(interior_tensor)  # shape (IG0*IG1, W)
    # compute the hidden space evaluated at the boundary
    _, H_bc_left = model(boundary_tensor_left)  # shape (Nbc, W)
    _, H_bc_right  = model(boundary_tensor_right)

    # compute the hidden space evaluated at the initial condition
    _, H_ic = model(initial_tensor)  # shape (Nic, W)
    # compute the gradients of the hidden H in the interior data points
    print("Differentiating H w.r.t. x now...")
    H2x = compute_H2x(H, x)  # of shape (IG0*IG1, W)
    print("Finished computing H2x.")

    print("Differentiating H w.r.t. t now...")
    Ht = compute_Ht(H, t)  # of shape (IG0*IG1, W)
    print("Finished computing Ht")

    H = H.detach().numpy()  # shape (IG0*IG1, W)
    H_ic = H_ic.detach().numpy()  # shape (Nic, W)
    H_bc_left = H_bc_left.detach().numpy()  # shape (Nbc, W)    
    H_bc_right = H_bc_right.detach().numpy()

    # now add another dimension to the H and Hx Ht if needed
    # all zeros for Hx and Ht and all ones for H
    if bias:
        H = np.hstack((H, np.ones((H.shape[0], 1))))
        H_ic = np.hstack((H_ic, np.ones((H_ic.shape[0], 1))))
        H_bc_left = np.hstack((H_bc_left, np.ones((H_bc_left.shape[0], 1))))
        H_bc_right = np.hstack((H_bc_right, np.ones((H_bc_right.shape[0], 1))))
        H2x = np.hstack((H2x, np.zeros((H2x.shape[0], 1))))
        Ht = np.hstack((Ht, np.zeros((Ht.shape[0], 1))))
    
    DH2x = D * H2x
    H_star = Ht - DH2x  # (I, W)(1/log['domain_info']['T'])*

    H_dict = {'H': H, 'H_ic': H_ic, 'H_bc_left': H_bc_left, 'H_bc_right': H_bc_right,
              'H2x': H2x, 'Ht': Ht,
              'IG': IG, 'Nic': Nic,'Nbc': Nbc,
              'DH2x': DH2x,'H_star': H_star}
    
    return H_dict


#  this function compute and inverts the matrix M
def compute_M(H_dict,w_pde,w_ic,w_bc):
    IG = H_dict['IG']

    Nic = H_dict['Nic']
    N_bc =  H_dict['Nbc']

    H_ic = H_dict['H_ic']
    H_bc_left = H_dict['H_bc_left']
    H_bc_right = H_dict['H_bc_right']
    N_i = IG[0] * IG[1]
    H_dict['N'] = N_i
    M = w_pde*(H_dict['H_star'].T @ H_dict['H_star']) / N_i + w_ic*(H_ic.T @ H_ic) / Nic + w_bc*(H_bc_left.T @ H_bc_left) / N_bc + w_bc*(H_bc_right.T @ H_bc_right) / N_bc# shape (W, W)
    
    Minv,_,_,_ = np.linalg.lstsq(M, np.eye(M.shape[0]))   
    H_dict['M'] = M
    H_dict['M_inv'] = Minv

    return M, Minv, H_dict



def compute_TL_with_F(forcing_function, w_pde, H_dict, input=None):
    '''
    if the forcing is a function, specify an input
    '''
    start_time = time.time()
    # forcing function
    if input is not None:
        forcing_value = forcing_function(input).reshape(-1, 1).cpu().detach().numpy()
        Rf = w_pde * (H_dict["H_star"].T @ forcing_value) / H_dict['N']
    else:
        Rf = w_pde * (H_dict["H_star"].T @ forcing_function.reshape(-1, 1)) / H_dict["N"]
    
    # compute W
    R = Rf + H_dict['R_ic'] + H_dict['R_bcs'] 
    W = H_dict['M_inv'] @ R  # shape 
    
    computational_time = time.time() - start_time
    return W, computational_time


def compute_Rf(H_dict, w_pde, forcing_function, input=None):
    '''
    if the forcing is a function, specify an input
    '''
    # forcing function
    if input is not None:
        forcing_value = forcing_function(input).reshape(-1, 1).cpu().detach().numpy()
        Rf = w_pde * (H_dict["H_star"].T @ forcing_value) / H_dict['N']
    else:
        Rf = w_pde * (H_dict["H_star"].T @ forcing_function.reshape(-1, 1)) / H_dict["N"]
    
    H_dict['R_f'] = Rf

    return Rf, H_dict


def compute_R_ic(H_dict, w_ic,log, ic_value=None, ic_function=None):

    H_ic_0 = H_dict['H_ic']
    
    
    N_ic = H_dict['Nic']

    # initial condition
    if ic_function is None:  # if we have constant initial condition
        Ric = w_ic*((ic_value * H_ic_0.T).sum(axis=1) / N_ic).reshape(-1, 1)
    else:  # if ic is a function

        _, _, initial_tensor = generate_initial_tensor(Nic=N_ic, x_span=(0,log['domain_info']['L']), require_grad=True, method='equally-spaced')
        forcing_ic = ic_function(initial_tensor).reshape(-1, 1).cpu().detach().numpy()
        Ric = w_ic * (H_ic_0.T @ forcing_ic) / N_ic
        
    H_dict['R_ic'] = Ric

    return Ric,H_dict

def compute_R_bcs(H_dict,w_bc, log, boundary_functions=None, boundary_values = None):
    '''
    boundary_values is a list with the bc on the at t=0 and t=T (left and right)
    '''
    
    H_bc_left = H_dict['H_bc_left']
    H_bc_right = H_dict['H_bc_right']
    
    
    N_bc = H_dict['Nbc']
    if boundary_functions is None:
        Rbc_left = w_bc*((boundary_values[0] * H_bc_left.T).sum(axis=1) / N_bc).reshape(-1, 1)
        Rbc_right = w_bc*((boundary_values[1] * H_bc_right.T).sum(axis=1) / N_bc).reshape(-1, 1)

    else:
        _, _, boundary_tensor_left = generate_boundary_tensor(Nbc=N_bc, t_span=(0,log['domain_info']['T']), x = 0, require_grad=True, method='equally-spaced')
        _, _, boundary_tensor_right = generate_boundary_tensor(Nbc=N_bc, t_span=(0,log['domain_info']['T']), x = log['domain_info']['L'], require_grad=True, method='equally-spaced')

        forcing_value_left = boundary_functions[0](boundary_tensor_left).reshape(-1, 1).cpu().detach().numpy()
        forcing_value_right = boundary_functions[1](boundary_tensor_right).reshape(-1, 1).cpu().detach().numpy()
        Rbc_left = w_bc * (H_bc_left.T @ forcing_value_left) / N_bc
        Rbc_right = w_bc * (H_bc_right.T @ forcing_value_right) / N_bc

    H_dict['R_bcs'] = Rbc_left + Rbc_right

    return Rbc_left, Rbc_right , Rbc_left + Rbc_right, H_dict

def compute_solution(H, W, IG):
    return (H @ W).reshape(IG, -1).T

def force_func_perturbation(n):
    solution_index = []
    for a in range(n+1):
        for b in range(a+1):
                if ((a+b)==n):
                    if (a==b):
                        solution_index.append([a, b, 1])
                    else:
                        solution_index.append([a, b, 2])
    return solution_index

def force_func_perturbation3(n):
    solution_index = [] # ind1, ind2, ind3, coeff
    for a in range(n+1):
        for b in range(a+1):
              for c in range(b+1):
                if ((a+b+c)==n):
                    if ((a==b) & (b==c)):
                        solution_index.append([a, b, c, 1])
                    elif ((a!=b) & (b!=c)):
                        solution_index.append([a, b, c, 6])
                    else:
                        solution_index.append([a, b, c, 3])
    return solution_index

######pertubation solution without change of IC and BC (same as computed with the TL method),used to test the homogeneous NN
def compute_perturbation_solution_with_F(p, epsilon, forcing, H_dict, input, training_log):
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
        else:
            force_function_index = force_func_perturbation(j-1)
            force_perturbation = 0
            for (a, b, coefficient) in force_function_index:
                force_perturbation -= coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0] 
            force_perturbation += perturbation_solution[j-1][:, 0]
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time

### for other polynomial perturbation, we can use the same function as above but with a different force function index
def compute_perturbation_solution_with_F3(p, epsilon, forcing, H_dict, input, training_log):
    '''
    for u(1-u^2)'''
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
        else:
            force_function_index = force_func_perturbation3(j-1)
            force_perturbation = 0
            for (a, b, c, coefficient) in force_function_index:
                force_perturbation -= coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0]*perturbation_solution[c][:, 0]
            force_perturbation += perturbation_solution[j-1][:, 0]
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time

def compute_perturbation_solution_with_F2(p, epsilon, forcing, H_dict, input, training_log):
    '''
    for u^2(1-u)'''
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
        else:
            force_function_index = force_func_perturbation3(j-1)
            force_perturbation = 0
            for (a, b, c, coefficient) in force_function_index:
                force_perturbation -= coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0]*perturbation_solution[c][:, 0]
            force_function_index = force_func_perturbation(j-1)
            for (a, b, coefficient) in force_function_index:
                force_perturbation += coefficient*perturbation_solution[a][:, 0]*perturbation_solution[b][:, 0]
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time

def compute_perturbation_solution_with_F_polynomial(p, epsilon, forcing, H_dict, input, training_log,Polynomial):
    '''
    for a general polynomial P : a list [[a_k,k]] where a_k is the coefficient of u^k (for non zero a_k)
    P is such that Du +epsilon*P(u) = F
    '''
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
        else:
            force_perturbation = 0
            for [p_k, k] in Polynomial:
                force_function_index = force_func_perturbation_monomial(j-1, k)
                for index_list in force_function_index:
                    term = perturbation_solution[index_list[0]][:, 0].copy()
                    for i in range(1, len(index_list) - 1):  # last element is coefficient
                        term *= perturbation_solution[index_list[i]][:, 0]
                    force_perturbation -= index_list[-1] * p_k * term

            
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time

def compute_perturbation_solution_polynomial(p, epsilon, forcing, H_dict, input, training_log,Polynomial,boundary_functions=None, boundary_values = None):
    '''
    for a general polynomial P : a list [[a_k,k]] where a_k is the coefficient of u^k (for non zero a_k)
    P is such that Du +epsilon*P(u) = F
    '''
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
        
            _,_,_,H_dict = compute_R_bcs(H_dict,boundary_functions=boundary_functions,boundary_values=boundary_values ,w_bc=training_log['w_bc'], log=training_log)
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)

            ### compute R_bc 0 for the next equations
            _,_,_,H_dict = compute_R_bcs(H_dict,boundary_functions=None,boundary_values=[0,0] ,w_bc=training_log['w_bc'], log=training_log)
        else:
            force_perturbation = 0
            for [p_k, k] in Polynomial:
                force_function_index = force_func_perturbation_monomial(j-1, k)
                for index_list in force_function_index:
                    term = perturbation_solution[index_list[0]][:, 0].copy()
                    for i in range(1, len(index_list) - 1):  # last element is coefficient
                        term *= perturbation_solution[index_list[i]][:, 0]
                    force_perturbation -= index_list[-1] * p_k * term

            
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time


def compute_perturbation_solution_polynomial_complete(p, epsilon, forcing, H_dict, input, training_log,Polynomial,boundary_functions=None, boundary_values = None,ic_function=None,ic_value=None):
    '''
    for a general polynomial P : a list [[a_k,k]] where a_k is the coefficient of u^k (for non zero a_k)
    P is such that Du +epsilon*P(u) = F
    '''
    perturbation_solution = []
    for j in range(p+1):
        if j==0:
            _,H_dict = compute_R_ic(H_dict,w_ic=10, log=training_log,ic_function=ic_function,ic_value=ic_value)
            
            _,_,_,H_dict = compute_R_bcs(H_dict,boundary_functions=boundary_functions,boundary_values=boundary_values ,w_bc=training_log['w_bc'], log=training_log)
            W, TL_time= compute_TL_with_F(forcing_function=forcing, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=input)
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)

            ### compute R_bc 0 for the next equations
            
            # _,_,_,H_dict = compute_R_bcs(H_dict,boundary_functions= None, boundary_values=[0,0] ,w_bc=training_log['w_bc'], log=training_log)
            # # _,H_dict = compute_R_ic(H_dict,w_ic=training_log['w_ic'], log=training_log,ic_value=0)
            H_dict['R_ic'] = np.zeros_like(H_dict['R_ic'])  # set the initial condition to 0 for the next equations
            H_dict['R_bcs'] = np.zeros_like(H_dict['R_bcs'])
        else:
            force_perturbation = 0
            for [p_k, k] in Polynomial:
                if k==0 and j==1: ## if the polynomial has a constant term, we need to add it only for the first perturbation equation
                    force_perturbation -= p_k
                if k!=0:    
                    force_function_index = force_func_perturbation_monomial(j-1, k)
                    for index_list in force_function_index:
                        term = perturbation_solution[index_list[0]][:, 0].copy()
                        for i in range(1, len(index_list) - 1):  # last element is coefficient
                            term *= perturbation_solution[index_list[i]][:, 0]
                        force_perturbation -= index_list[-1] * p_k * term

            
            W, time = compute_TL_with_F(forcing_function=force_perturbation, w_pde=training_log["w_pde"],
                                                          H_dict=H_dict, input=None) ###Now input is none because we treat force pertubation as an array with the value of the function
            perturbation_solution.append(compute_solution(H_dict['H'], W, H_dict['N']).T)
            TL_time += time
    NN_TL_solution = sum([(epsilon**k)*perturbation_solution[k] for k in range(p+1)])
    return NN_TL_solution, TL_time


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
    return [list(tup) for tup in combinations_with_replacement(range(N), num_indices) if sum(tup) == N - 1]
            
def force_func_perturbation_monomial(n, power):

    index_list = index_tuples(n+1, power)
    solution_index = [index_list[i][::-1] + [number_combinations(index_list[i])] for i in range(len(index_list))][::-1]
    return solution_index

def number_combinations(index_list: List[int]) -> int:
    counts = Counter(index_list)
    denom = 1
    for count in counts.values():
        denom *= factorial(count)

    return factorial(len(index_list)) // denom