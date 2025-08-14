# Perturbation-Guided Transfer Learning with Physics-Informed Neural Networks for Nonlinear Systems: PTL-PINN

**Author:** Duarte Alexandrino  
**Supervisors:** Prof. Pavlos Protopapas, Dr. Ben Moseley  
**MSc in Applied Computational Science and Engineering — Imperial College London**  
**Contact:** d.alexandrino2010@gmail.com

> Master’s thesis project proposing fast, accurate framework to solve weakly nonlinear oscillators by combining perturbation theory with one-shot transfer learning in Multi-Headed PINNs.

---

## 🔎 Overview

Accurately solving nonlinear differential equations is central to modeling real-world dynamical systems. **Perturbation methods** approximate weakly nonlinear systems via a hierarchy of simpler linear problems, providing quantitative accuracy and qualitative insight.

This repository contains the code used to develop the **PTL-PINN** — a **Perturbation-Guided Transfer Learning** framework for **Physics-Informed Neural Networks (PINNs)**. By training foundational PINN models on families of linear ODEs representative of the perturbation system and reusing a shared latent representation, PTL-PINN can reconstruct perturbation series efficiently with one-shot transfer learning.

---

## Key Contributions

- **Novel Lindstedt-Poincare implementation**: new numerical and scalable implementation of the Lindstedt-Poincare for undamped nonlinear oscillators with polynomial nonlinearity
- **Foundational PINNs models**: Multi-Headed-PINNs trained for **undamped**, **underdamped**,and **overdamped** regimes with Fourier features and sinusoidal activations to mitigate spectral bias.
- **Evaluation of the pratical applicability of perturbation methods**: exploring resonance/near-resonance, convergence of the frequency series, and practical truncation criteria.
- **Evaluation of one-shot transfer learning**: proposing and demonstrating the reuse of precomputed matrix $M^{-1}$.
- **Performance vs. classical solvers (RK45, Radau)** demonstrating comparable accuracy and up to **10×** faster inference.


---

## Repository Structure

```
PTL-PINNs/
├── ptlpinns/                          
│   ├── __init__.py
│   │
│   ├── odes/   
│   │   ├── __init__.py
│   │   ├── numerical.py        # numerical solver
│   |   ├── forcing.py          # forcing functions      
│   |   ├── equations.py        # ODE definition
│   |   └── plot.py             # plotting 
│   |
│   ├── models/
│   │   ├── __init__.py
│   |   ├── model.py            # model architecture
│   |   ├── training.py         # training functions
│   |   ├── one_shot.py         # one-shot
│   |   └── transfer.py         # transfer logic
│   |   └── train/              # models training
│   |      ├── config/
│   |      |    ├── undamped.yml
│   |      |    ├── underdamped.yml
│   |      |    └── overdamped.yml
│   |      |
│   |      ├── undamped.ipynb
│   |      ├── underdamped.ipynb
│   |      └── overdamped.ipynb
│   |
│   └── perturbation/
│   │   ├── __init__.py
│   │   ├── standard.py         # standard perturbation
│   │   └── LPM.py              # Lindstedt-Poincare method
│   │
│   └── results/
│       ├── __init__.py
│       ├── undamped.ipynb          # standard vs Lindstedt-Poincare
│       ├── lpm_forcing.ipynb       # LPM forcing multiple passes
│       ├── underdamped.ipynb       # underdamped results (and near-resonance)
│       ├── overdamped.ipynb        # overdamped results (and ic blow-up)
│       └── time.ipynb              #  timings: classic solvers vs. PTL-PINNs         
│
├── figures/                        # figures presented in the README.md
├── pyproject.toml          
└── README.md
```

---

## Multi-Headed-PINN architecture
 
Multi-Headed-PINN uses a shared latent representation to approximate equations of
similar form. It maps time to a latent representation that is used when inferring for a new parameter regime. To mitigate the spectral bias observed when training oscillatory solutions, we use Fourier feature embeddings at the input layer sinusoidal activation functions.

![PTL-PINN architecture](figures/MH-PINN.png)

---

## Training equations

We train a different model for the undamped, underdamped and overdamped regimes. The following figures present the linear differential equations used in training for each damping regime:

### Undamped 

![Undamped training](figures/undamped_training.png)

### Underdamped

![Underdamped training](figures/underdamped_training.png)

### Overdamped

![Overdamped training](figures/overdamped_training.png)

---
