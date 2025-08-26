### One-shot transfer learning: Pytorch version

The following code presents a discard version of one-shot transfer learning, where all the variables are tensor. This code was developed so that the second derivatives of Lindstedt-Poincare could be computed using torch.autograd. This version is not used for the following reasons:

- computational graph overhead: tracking all matrices and vector needed for the computation of a perturbation correction is very expensive. one-shot transfer learning does a lot of operations on different matrices.

- ill-conditioned matrix: the parameters in H_dict are all the same when compared to the original (numpy) version of this code. However, the matrix inversion adds significant differences.