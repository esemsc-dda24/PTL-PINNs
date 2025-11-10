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
          pde_weight=1, bc_weight=1, ic_weight=1, scheduler=None, method='equally-spaced-noisy'):

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

#######################Plotting functions########################
def plot_loss(loss_trace, pde_trace, bc_trace, ic_trace, data_trace=None, path=None):
    num_iter = len(loss_trace)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex=True, layout='constrained')
    ax[0].plot(range(1, num_iter + 1), loss_trace, label='Total Loss')
    ax[1].plot(range(1, num_iter + 1), pde_trace, label='PDE Loss')
    ax[1].plot(range(1, num_iter + 1), ic_trace, label="IC Loss")
    ax[1].plot(range(1, num_iter + 1), bc_trace, label="BC Loss")
    if data_trace:
        ax[1].plot(range(1, num_iter + 1), data_trace, label="Data Loss")
    ax[0].semilogy()
    ax[0].xaxis.set_tick_params(labelsize=11)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].set_xlabel("Number of iterations", fontsize=12)
    ax[0].set_title("Total Loss Value vs. Iteration", fontsize=14)
    ax[0].grid()
    ax[0].legend(loc="best", fontsize=11)

    ax[1].semilogy()
    ax[1].xaxis.set_tick_params(labelsize=11)
    ax[1].yaxis.set_tick_params(labelsize=12)
    ax[1].set_xlabel("Number of iterations", fontsize=12)
    ax[1].set_title("PDE and IC Loss Value vs. Iteration", fontsize=14)
    ax[1].grid()
    ax[1].legend(loc="best", fontsize=11)

    fig.supylabel("Loss", fontsize=18)
    if path is not None:
        plt.savefig(path)


def plot_solution1(solution, mesh_x, mesh_t, surface=True, path=None, title=None, rotation=(25, -60)):
    """
    args:
        solution (np.ndarray): shape (Nx, Nt) or (Nt, Nx)
        mesh_x, mesh_t (np.ndarray): meshgrids of size (Nx, Nt)
        surface (bool): True for 3D surface plot, False for contourf
        path (str): path to save the figure
        title (str): title of the plot
        rotation (tuple): (elev, azim) for 3D view
    
    """
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d' if surface else None)
    solution = solution.swapaxes(0, 1)  # shape (Nx, Nt)
    if surface:
        ax.plot_surface(mesh_x, mesh_t, solution, cmap='viridis')
        ax.view_init(rotation[0], rotation[1])
        ax.set_xlabel("$x$", fontsize=13)
        ax.set_ylabel("$t$", fontsize=13)
        ax.set_zlabel("u", fontsize=13)
    else:
        min_level = solution.min() ##if ((solution.min() > 0) and (solution.min() < 1e-10)) else 0.
        levels = np.linspace(min_level, solution.max(), 20)
        Cs = ax.contourf(mesh_x, mesh_t, solution, levels=levels, extend="min")
        cbar = fig.colorbar(Cs)
        new_ticks = np.linspace(min_level, solution.max(), 20)[::2]
        cbar.set_ticks(new_ticks)
        cbar.ax.tick_params(labelsize=12)
        ax.set_xlabel("$x$", fontsize=16)
        ax.set_ylabel("$t$", fontsize=16)

    ax.set_title(title if title is not None else "PINN solution", fontsize=16)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.show()



def plot_4_heads(NN_solution, mesh_x, mesh_t, surface=True, title_prefix="Tête", path=None, rotation=(25, -60)):
    """
    Affiche les sorties 3D ou en contours pour les 4 têtes d’un PINN.
    Args:
        NN_solution (np.ndarray): shape (4, Nx, Nt)
        mesh_x, mesh_t (np.ndarray): meshgrids de taille (Nx, Nt)
        surface (bool): True pour 3D surface, False pour contourf
        title_prefix (str): texte à mettre dans les titres
        path (str): chemin pour sauvegarde (facultatif)
        rotation (tuple): (elev, azim) pour la vue 3D
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                             subplot_kw={'projection': '3d'} if surface else {},
                             tight_layout=True)
    NN_solution = NN_solution.swapaxes(1, 2)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(4):
        Z = NN_solution[i, :, :]
        ax = axes[i]

        if surface:
            ax.plot_surface(mesh_x, mesh_t, Z, cmap='viridis')
            ax.view_init(elev=rotation[0], azim=rotation[1])
        else:
            min_level = Z.min() if (0 < Z.min() < 1e-10) else 0.
            levels = np.linspace(min_level, Z.max(), 20)
            Cs = ax.contourf(mesh_x, mesh_t, Z, levels=levels, extend="min")
            plt.colorbar(Cs, ax=ax, shrink=0.8)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        if surface:
            ax.set_zlabel("$u$")
        ax.set_title(f"{title_prefix} {i}")

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    plt.show()

def plot_6_heads(NN_solution, mesh_x, mesh_t, surface=True, title_prefix="Tête", path=None, rotation=(25, -60)):
    """
    Affiche les sorties 3D ou en contours pour les 4 têtes d’un PINN.
    Args:
        NN_solution (np.ndarray): shape (4, Nx, Nt)
        mesh_x, mesh_t (np.ndarray): meshgrids de taille (Nx, Nt)
        surface (bool): True pour 3D surface, False pour contourf
        title_prefix (str): texte à mettre dans les titres
        path (str): chemin pour sauvegarde (facultatif)
        rotation (tuple): (elev, azim) pour la vue 3D
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10),
                             subplot_kw={'projection': '3d'} if surface else {},
                             tight_layout=True)
    NN_solution = NN_solution.swapaxes(1, 2)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(6):
        Z = NN_solution[i, :, :]
        ax = axes[i]

        if surface:
            ax.plot_surface(mesh_x, mesh_t, Z, cmap='viridis')
            ax.view_init(elev=rotation[0], azim=rotation[1])
        else:
            min_level = Z.min() if (0 < Z.min() < 1e-10) else 0.
            levels = np.linspace(min_level, Z.max(), 20)
            Cs = ax.contourf(mesh_x, mesh_t, Z, levels=levels, extend="min")
            plt.colorbar(Cs, ax=ax, shrink=0.8)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$t$")
        if surface:
            ax.set_zlabel("$u$")
        ax.set_title(f"{title_prefix} {i}")

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    plt.show()

def load_pretrain_model(model_name, mesh_grid, Nx, Nt):
    
    model_path = os.path.join("training_log", model_name, "model.pickle")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    log_path = os.path.join("training_log", model_name, "log.pickle")
    with open(log_path, 'rb') as f:
        log = pickle.load(f)

    NN_solution, _ = model(mesh_grid.cpu())
    NN_solution = NN_solution.cpu().detach().numpy()
    NN_solution = np.swapaxes(NN_solution, 2, 1).reshape(-1,Nx, Nt)
    

    return model, log, NN_solution

######################functions for forcing functions########################

def forcing_function_constant(input):
    return torch.ones((input.shape[0], 1), dtype=torch.double) #forcing f=1
    # return torch.zeros((input.shape[0], 1), dtype=torch.double) #forcing f=0

def forcing_function_t(input):
    t = input[:, 1].unsqueeze(1)  
    return t

def forcing_function_t2(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 2

def forcing_function_t3(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 3

def forcing_function_t4(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 4

def forcing_function_t5(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 5

def forcing_function_t6(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 6   

def forcing_function_t7(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 7

def forcing_function_t9(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 9

def forcing_function_t11(input):
    t = input[:, 1].unsqueeze(1)
    return t ** 11  

def forcing_sincos(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.sin(t)*torch.cos(x)

def forcing_sin2cos3(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.sin(2*t)*torch.cos(3*x)

def forcing_xt(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return x * t

def forcing_exponential(input):
    t = input[:, 1].unsqueeze(1)
    x = input[:, 0].unsqueeze(1)
    return torch.exp(-0.1*((t-0.5*torch.ones_like(t)**2) + (x - 0.5*torch.ones_like(x)**2)))  


def forcing_atan(input):
    t = input[:, 1].unsqueeze(1)
    x = input[:, 0].unsqueeze(1)
    return torch.arctan(t + x)  

def forcing_sinxt(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.sin(x + t)

def forcing_xt2(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return x ** 2 + t ** 2

def forcing_xexp(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.exp(-t)*x

def forcing_zeros(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.zeros_like(x)  # Example forcing function


def bc_exp(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.exp(-t)

def bc_sin(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.sin(- t)  # Example boundary condition function


def initial_step(input):
    N = input.shape[0]
    return torch.cat([torch.ones(N//2),torch.zeros(N//2)]) 

def initial_condition0(input):
    return torch.zeros((input.shape[0], 1), dtype=torch.double)     ## initial condition u(x,0)=0

def ic_sin(input):
    x = input[:, 0].unsqueeze(1)
    t = input[:, 1].unsqueeze(1)
    return torch.sin(x)  # Example boundary condition function

def ic_stepexp1(input):
    '''1/(1+torch.exp(10*(x-1)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(10*(x-1)))  # Example initial condition function

def ic_stepexp125(input):
    '''1/(1+torch.exp(10*(x-1.25)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(10*(x-1.25)))  # Example initial condition function

def ic_stepexp075(input):
    '''1/(1+torch.exp(10*(x-0.75)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(10*(x-0.75)))  # Example initial condition function

def ic_stepexp05(input):
    '''1/(1+torch.exp(10*(x-0.5)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(20*(x-0.5)))  # Example initial condition function

def ic_stepexp09(input):
    '''1/(1+torch.exp(10*(x-0.9)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(10*(x-0.9)))  # Example initial condition function

def ic_stepexp15(input):
    '''1/(1+torch.exp(10*(x-1.5)))
    '''
    x =input[:, 0].unsqueeze(1)
    return 1/(1+torch.exp(10*(x-1.5)))  # Example initial condition function


def echelon_square(x_min, x_max, val,sigma=10):
    
    def forcing_function_echelon(input):
        x = input[:, 0].unsqueeze(1)
        return val*(1/(1+torch.exp(-sigma*(x-x_min))) + 1/(1+torch.exp(sigma*(x-x_max)))-1)
    
    return forcing_function_echelon


def constant_function(constant):
    '''return a function input  ((x,t)in torch) -> constant, can be given as function.

    for example constant_function(0) will return a function that returns 0 for any input.
    '''
    def forcing_function_constant(input):
        return constant*torch.ones_like(input[:, 0].unsqueeze(1))
    return forcing_function_constant
