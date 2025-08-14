import torch
from neurodiffeq import diff
from neurodiffeq.generators import Generator1D
import numpy as np
from matplotlib import pyplot as plt

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