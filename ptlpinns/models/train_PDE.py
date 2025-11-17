import torch
import torch.nn as nn
import torch.nn.functional as F
from neurodiffeq import diff
from neurodiffeq.generators import Generator2D,Generator1D
import numpy as np
from matplotlib import pyplot as plt
import time
from tqdm.auto import tqdm
import pickle
import os

def loss_KPP(model, interior_grid, x_span, t_span, Nic, Nbc,
         boundary_values, initial_value, Forcing_functions,
         D, pde_weight=1, bc_weight=1, ic_weight=1, method='equally-spaced-noisy'):
    """
    Calculate the loss function for the KPP-Fisher equation. Refer to:

        - residual = dudt - D*d2udx2 - force  
        - pde_loss = F.mse_loss(residual, torch.zeros_like(residual))
    """

    generator = Generator2D(grid=interior_grid, method=method,  xy_min=(x_span[0], t_span[0]), xy_max=(x_span[1], t_span[1]))
    samples = generator.get_examples()

    x_input = samples[0].unsqueeze(1).double().requires_grad_(True) # shape (N, 1)
    t_input = samples[1].unsqueeze(1).double().requires_grad_(True) # shape (N, 1)
    input = torch.cat([x_input, t_input], dim=1)  # shape (N, 2)

    output, _ = model(input)  # shape (k,N,1)
    u = output.squeeze(-1).T  # (N, k)

    # compute the PDE residual
    dudt = torch.cat([diff(u[:, i].reshape(-1, 1), t_input) for i in range(u.shape[1])], dim=1)  # shape (N, k)
    d2udx2 = torch.cat([diff(u[:, i].reshape(-1, 1), x_input, order=2) for i in range(u.shape[1])], dim=1)  # shape (N, k)
    force = torch.cat([Forcing_functions[i](input) for i in range(len(Forcing_functions))], dim=1)  # (N, k)

    residual = dudt - D*d2udx2 - force  
    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    t_boundary_sample = Generator1D(size=Nbc, method='equally-spaced', t_min=t_span[0], t_max=t_span[1]).get_examples()
    x_boundary_left = torch.zeros_like(t_boundary_sample).unsqueeze(1).requires_grad_()
    x_boundary_right =x_span[1]*torch.ones_like(t_boundary_sample).unsqueeze(1).requires_grad_()
    t_boundary = t_boundary_sample.unsqueeze(1).requires_grad_()
    
    input_boundary_left = torch.cat([x_boundary_left, t_boundary], dim=1).double()  # (Nbc, 2)
    input_boundary_right = torch.cat([x_boundary_right, t_boundary], dim=1).double()  # (Nbc, 2)
    # evaluate the neural network at the boundary
    output_boundary_left, _ = model(input_boundary_left)  # (k, Nbc, 1)
    output_boundary_right, _ = model(input_boundary_right)  # (k, Nbc, 1)
    
    boundary_val = torch.tensor(np.array(boundary_values), dtype=torch.double)
    k = len(boundary_values)
    
    left_truth = boundary_val[:,0].view(k,1,1).expand(k,Nbc,1)  # left boundary value
    right_truth = boundary_val[:,1].view(k,1,1).expand(k,Nbc,1)  # right boundary value
    bc_loss = F.mse_loss(output_boundary_left, left_truth) + F.mse_loss(output_boundary_right, right_truth)

    # initial conditions
    x_initial_samples = Generator1D(size=Nic, method='equally-spaced', t_min=x_span[0], t_max=x_span[1]).get_examples()
    x_initial = torch.cat([
        x_initial_samples,
    ]).unsqueeze(1)
    t_initial = torch.cat([
        torch.zeros_like(x_initial_samples),
    ]).unsqueeze(1)
    x_initial.requires_grad_()
    t_initial.requires_grad_()
    input_initial = torch.cat([x_initial, t_initial], dim=1)  # (N_initial, 2)
    # evaluate the neural network at the initial conditions
    output_initial, _ = model(input_initial)  # (k, N_initial, 2)

    if isinstance(initial_value, (int, float)):
        truth_initial = torch.ones_like(output_initial) * initial_value
    elif isinstance(initial_value[0], (int, float)):
        truth_initial = torch.ones_like(output_initial) * torch.tensor(np.array(initial_value)[:, np.newaxis])
        truth_initial.to(output_initial.device)
    # if the initial_value is a list of functions
    else:
        truth_initial = torch.stack([truth(input_initial) for truth in initial_value])  # (k, N_initial)
    
    ic_loss = F.mse_loss(output_initial, truth_initial)

    # sum the weighted loss
    total_loss = pde_weight * pde_loss + ic_weight * ic_loss + bc_weight * bc_loss

    return total_loss, pde_loss, ic_loss, bc_loss, output[-1, ...]


def loss_wave(model, interior_grid, x_span, t_span, Nic, Nbc,
         boundary_values, initial_value,initial_Neumann, Forcing_functions,
         c, pde_weight=1, bc_weight=1, ic_weight=1, method='equally-spaced-noisy'):
    """
    Calculates the residuals of the Wave equation. Refer to:

        - residual = d2udt2 - (c**2)*d2udx2 - force  
        - pde_loss = F.mse_loss(residual, torch.zeros_like(residual))
    """

    generator = Generator2D(grid=interior_grid, 
                            method=method, 
                        xy_min=(x_span[0], t_span[0]), xy_max=(x_span[1], t_span[1]))

    samples = generator.get_examples()
    x_input = samples[0].unsqueeze(1).double().requires_grad_(True)
    t_input = samples[1].unsqueeze(1).double().requires_grad_(True)
    
    # concatenate the input to the network
    input = torch.cat([x_input, t_input], dim=1)  # shape (N, 2)
    
    # evaluate the network on these points
    output, _ = model(input)  # shape (k,N,1)
    u = output.squeeze(-1).T  # (N, k)

    # # compute the gradients
    d2udt2 = torch.cat([diff(u[:, i].reshape(-1, 1), t_input, order=2) for i in range(u.shape[1])], dim=1)  # shape (N, k)
    d2udx2 = torch.cat([diff(u[:, i].reshape(-1, 1), x_input, order=2) for i in range(u.shape[1])], dim=1)  # shape (N, k)
    force = torch.cat([Forcing_functions[i](input) for i in range(len(Forcing_functions))], dim=1)  # (N, k)
    residual = d2udt2 - (c**2)*d2udx2 - force  
    pde_loss = F.mse_loss(residual, torch.zeros_like(residual))

    t_boundary_sample = Generator1D(size=Nbc, method='equally-spaced', t_min=t_span[0], t_max=t_span[1]).get_examples()
    x_boundary_left = torch.zeros_like(t_boundary_sample).unsqueeze(1).requires_grad_()
    x_boundary_right =x_span[1]*torch.ones_like(t_boundary_sample).unsqueeze(1).requires_grad_()
    t_boundary = t_boundary_sample.unsqueeze(1).requires_grad_()

    
    input_boundary_left = torch.cat([x_boundary_left, t_boundary], dim=1).double()  # (Nbc, 2)
    input_boundary_right = torch.cat([x_boundary_right, t_boundary], dim=1).double()  # (Nbc, 2)
    # evaluate the neural network at the boundary
    output_boundary_left, _ = model(input_boundary_left)  # (k, Nbc, 1)
    output_boundary_right, _ = model(input_boundary_right)  # (k, Nbc, 1)
    
    boundary_val = torch.tensor(np.array(boundary_values), dtype=torch.double)
    k = len(boundary_values)
    
    left_truth = boundary_val[:,0].view(k,1,1).expand(k,Nbc,1)  # left boundary value
    right_truth = boundary_val[:,1].view(k,1,1).expand(k,Nbc,1)  # right boundary value

    bc_loss = F.mse_loss(output_boundary_left, left_truth) + F.mse_loss(output_boundary_right, right_truth)

    x_initial_samples = Generator1D(size=Nic, method='equally-spaced', t_min=x_span[0], t_max=x_span[1]).get_examples()
    x_initial = torch.cat([x_initial_samples,]).unsqueeze(1)
    t_initial = torch.cat([torch.zeros_like(x_initial_samples),]).unsqueeze(1)
    x_initial.requires_grad_()
    t_initial.requires_grad_()

    input_initial = torch.cat([x_initial, t_initial], dim=1)  # (N_initial, 2)
    # evaluate the neural network at the initial conditions
    output_initial, _ = model(input_initial)  # (k, N_initial,1)

    if isinstance(initial_value, (int, float)):
        truth_initial = torch.ones_like(output_initial) * initial_value
    elif isinstance(initial_value[0], (int, float)):
        truth_initial = torch.ones_like(output_initial) * torch.tensor(np.array(initial_value)[:, np.newaxis])
        truth_initial.to(output_initial.device)
    # if the initial_value is a list of functions
    else:
        truth_initial = torch.stack([truth(input_initial) for truth in initial_value])  # (k, N_initial)
    
    u_ic = output_initial.squeeze(-1).T 
    du_icdt = torch.cat([diff(u_ic[:, i].reshape(-1, 1), t_initial) for i in range(u.shape[1])], dim=1)  # shape (N, k)

    if isinstance(initial_Neumann, (int, float)):
        truth_initialN = torch.ones_like(output_initial) * initial_value
    elif isinstance(initial_Neumann[0], (int, float)):
        truth_initialN = torch.ones_like(output_initial) * torch.tensor(np.array(initial_Neumann)[:, np.newaxis])
        truth_initialN.to(output_initial.device)
    # if the initial_value is a list of functions
    else:
        truth_initialN = torch.stack([truth(input_initial) for truth in initial_Neumann])  # (k, N_initial)

    ic_loss = F.mse_loss(output_initial, truth_initial) + F.mse_loss(du_icdt, truth_initialN)  # (k, N_initial,1)
    total_loss = pde_weight * pde_loss + ic_weight * ic_loss + bc_weight * bc_loss

    return total_loss, pde_loss, ic_loss, bc_loss, output[-1, ...]


def train(model, optimizer, num_iter, Forcing_functions, boundary_values, initial_value,
          coeff, equation, interior_grid=(30, 30), x_span=(0, 1), initial_Neumann = None, 
          t_span=(0, 1), Nic=100, Nbc=100, every=100, save_epoch=None,
          pde_weight=1, bc_weight=1, ic_weight=1, scheduler=None, method='equally-spaced-noisy', epsilon = None):

    loss_trace = []
    pde_loss_trace = []
    ic_loss_trace = []
    bc_trace = []
    output_trace = {}
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()
        # evaluate the loss

        if equation == "Wave":
            total, pde, ic, bc, output_epoch = loss_wave(model=model, interior_grid=interior_grid,
                                                      x_span=x_span, t_span=t_span, Nic=Nic, Nbc=Nbc,
                                                      boundary_values=boundary_values, initial_value=initial_value,
                                                      initial_Neumann=initial_Neumann,
                                                      Forcing_functions=Forcing_functions,
                                                      c=coeff, pde_weight=pde_weight, bc_weight=bc_weight,
                                                      ic_weight=ic_weight, method=method)

        elif equation == "KPP-Fisher":
            total, pde, ic, bc, output_epoch = loss_KPP(model=model, interior_grid=interior_grid,
                                                      x_span=x_span, t_span=t_span, Nic=Nic, Nbc=Nbc,
                                                      boundary_values=boundary_values, initial_value=initial_value,
                                                      Forcing_functions=Forcing_functions,
                                                      D=coeff, pde_weight=pde_weight, bc_weight=bc_weight,
                                                      ic_weight=ic_weight, method=method)
        elif equation == "KPP-Fisher_nonlinear":
            pass
        else:
            raise ValueError("input a valid equation name")
    
    
        # take the step
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # record
        loss_trace.append(total.item())
        pde_loss_trace.append(pde.item() * pde_weight)
        ic_loss_trace.append(ic.item() * ic_weight)
        bc_trace.append(bc.item() * bc_weight)
        if save_epoch is not None:
            if (i + 1) % save_epoch == 0:
                output_trace[i + 1] = output_epoch.detach().cpu().numpy()

        if (i + 1) % every == 0:
            print(
                f"[{i + 1:6d}] Iter | "
                f"Total: {total.item():.4e} | "
                f"PDE: {pde.item():.4e} (w={pde_weight}) | "
                f"IC: {ic.item():.4e} (w={ic_weight}) | "
                f"BC: {bc.item():.4e} (w={bc_weight})"
            )

    return loss_trace, pde_loss_trace, ic_loss_trace,bc_trace, output_trace



# this function generate interior tensors to evaluate the Hs
# IG is the number of samples in each dimension
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