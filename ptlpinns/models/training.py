from tqdm.auto import tqdm
from neurodiffeq.generators import Generator1D
from neurodiffeq import diff
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from neurodiffeq.generators import Generator1D

def generate_eval_tensor(N=512, t_span=(0, 1), require_grad=True):
    generator = Generator1D(size=N, method="equally-spaced", t_min=t_span[0], t_max=t_span[1])
    t = generator.get_examples().unsqueeze(1)  # (N, 1)
    # convert this sample points into input to the network and requires gradients
    t = t.cpu()
    if require_grad:
        t.requires_grad_()
    return t

def loss(model, N, t_span, equation_functions, initial_condition_functions,
         forcing_functions, ode_weight=1, ic_weight=1, method='equally-spaced-noisy'):
    # use 1d equally-spaced-noisy generator to generate sample points
    generator = Generator1D(size=N, method=method, t_min=t_span[0], t_max=t_span[1])
    t = generator.get_examples().double()
    t[t < t_span[0]] = t_span[0]
    t[t > t_span[1]] = t_span[1]
    # requires gradients
    t.requires_grad_()
    input_tensor = t.unsqueeze(1)

    # evaluate the network on these points
    output_tensor, _ = model(input_tensor)  # shape (k, N, 2)

    # separate the x , y parts of the output of the network
    y1 = output_tensor[:, :, 0].T  # shape (N, k)
    y2 = output_tensor[:, :, 1].T  # shape (N, k)

    # compute the gradients
    dy1dt = torch.cat([diff(y1[:, i].reshape(-1, 1), input_tensor) for i in range(y1.shape[1])], dim=1)  # shape (N, k)
    dy2dt = torch.cat([diff(y2[:, i].reshape(-1, 1), input_tensor) for i in range(y2.shape[1])], dim=1)  # shape (N, k)

    # compute the equation part on N data points across k heads
    equation = torch.stack([equation_functions[i](y1[..., i], y2[..., i]) for i in range(len(equation_functions))], dim=1)  # (N, k, 2)

    # compute the forcing function on N data points across k heads
    force = torch.stack([forcing_functions[i](t) for i in range(len(forcing_functions))], dim=1)  # (N, k, 2)

    # compute the ode residual
    residual = torch.cat(
        [
            (dy1dt + equation[..., 0] - force[..., 1]).unsqueeze(2),
            (dy2dt + equation[..., 1] - force[..., 0]).unsqueeze(2),
        ],
        dim=2
    )
    # shape (N, k, 2)
    # compute the ode loss
    ode_loss = F.mse_loss(residual, torch.zeros_like(residual))

    # initial conditions
    t_initial = Generator1D(size=1, method='equally-spaced', t_min=0, t_max=0).get_examples()
    t_initial = t_initial.unsqueeze(0) # (1, 1)
    # evaluate the neural network at the initial conditions
    output_initial, _ = model(t_initial)  # (k, 1, 2)

    truth_initial = torch.stack([initial_condition_functions[i](t_initial.squeeze(0)) for i in range(len(initial_condition_functions))], dim=0)  # (k, 1, 2)

    ic_loss = F.mse_loss(output_initial, truth_initial)

    # sum the weighted loss
    total_loss = ode_weight * ode_loss + ic_weight * ic_loss

    return total_loss, ode_loss, ic_loss, output_tensor[-1, ...]


def train(model, optimizer, num_iter, equation_functions, initial_condition_functions, forcing_functions,
           N=512, t_span=(0, 1), every=100, save_epoch=None,
           ode_weight=1, ic_weight=1, scheduler=None, method='equally-spaced-noisy'):

    loss_trace = []
    ode_loss_trace = []
    ic_loss_trace = []
    output_trace = {}
    for i in tqdm(range(num_iter)):
        optimizer.zero_grad()
        # evaluate the loss
        total, ode, ic, output_epoch = loss(model=model, N=N, t_span=t_span, equation_functions=equation_functions,
                                            initial_condition_functions=initial_condition_functions, forcing_functions=forcing_functions, 
                                            ode_weight=ode_weight, ic_weight=ic_weight, method=method)
        # take the step
        total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # record
        loss_trace.append(total.item())
        ode_loss_trace.append(ode.item() * ode_weight)
        ic_loss_trace.append(ic.item() * ic_weight)
        if save_epoch is not None:
            if (i + 1) % save_epoch == 0:
                output_trace[i + 1] = output_epoch.detach().cpu().numpy()
        # print
        if (i + 1) % every == 0:
            print("{}th Iter: total {}, ode {}, ic {}".format(i + 1, total.item(), ode.item() * ode_weight, ic.item() * ic_weight))
    return loss_trace, ode_loss_trace, ic_loss_trace, output_trace

def plot_loss(loss_trace, ode_trace, ic_trace, path=None):
    num_iter = len(loss_trace)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex=True, layout='constrained')
    ax[0].plot(range(1, num_iter + 1), loss_trace, label='Total Loss')
    ax[1].plot(range(1, num_iter + 1), ode_trace, label='ODE Loss')
    ax[1].plot(range(1, num_iter + 1), ic_trace, label="IC Loss")

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
    ax[1].set_title("ODE and IC Loss Value vs. Iteration", fontsize=14)
    ax[1].grid()
    ax[1].legend(loc="best", fontsize=11)


    fig.supylabel("Loss", fontsize=18)
    if path is not None:
        plt.savefig(path)

def compare_training_solutions(NN_solution, linear_solution_list, t_eval, title_list, k):
    fig, axes = plt.subplots(k, 2, figsize=(10, 3 * k), sharex=True)

    for index in range(k):
        # Left column: function comparison
        axes[index, 0].plot(t_eval, NN_solution[index][:, 0], label="x(0) PINN", color="blue")
        axes[index, 0].plot(t_eval, linear_solution_list[index][0], label="x(0) RK45", color="orange", linestyle="--")
        axes[index, 0].set_title(f"{title_list[index]} - x")
        axes[index, 0].legend()
        axes[index, 0].grid(True)

        # Right column: derivative comparison
        axes[index, 1].plot(t_eval, NN_solution[index][:, 1], label="x'(0) PINN", color="green")
        axes[index, 1].plot(t_eval, linear_solution_list[index][1], label="x'(0) RK45", color="red", linestyle="--")
        axes[index, 1].set_title(f"{title_list[index]} - x'")
        axes[index, 1].legend()
        axes[index, 1].grid(True)

    plt.tight_layout()
    plt.show()