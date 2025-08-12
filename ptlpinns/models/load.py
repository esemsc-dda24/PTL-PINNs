import yaml
import numpy as np
from typing import Dict, List, Any
import torch

def initial_condition(y1_0, y2_0):
    def ic(t):
        return torch.stack((y1_0 * torch.ones_like(t), y2_0 * torch.ones_like(t)), dim=1)
    return ic

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"Value must be a list or tuple, got {type(x)}.")

def ptl_config(path: str) -> Dict[str, Any]:
    """
    Load PTL-PINN experiment config from YAML and expand it into parallel lists.
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    heads = cfg.get("heads", [])
    k = int(cfg.get("k", len(heads)))
    if len(heads) != k:
        raise ValueError(f"Mismatch: k={k} but len(heads)={len(heads)}.")

    rng = np.random.default_rng(cfg.get("seed"))

    w_list: List[float] = []
    zeta_list: List[float] = []
    mu_list: List[float] = []
    forcing_freq: List[List[float]] = []
    forcing_coef: List[List[float]] = []
    ic_python_list: List[List[float]] = []

    for i, h in enumerate(heads):
        w0   = float(h.get("w0"))
        zeta = float(h.get("zeta"))
        ic   = h.get("ic")

        if not (isinstance(ic, (list, tuple)) and len(ic) == 2):
            raise ValueError(f"Head {i}: 'ic' must be a 2-item list [x0, v0].")

        mu = h.get("mu")
        forcing = h.get("forcing")
        freq = _as_list(forcing.get("freq"))
        coef = forcing.get("coef", [])

        is_random = isinstance(coef, str) and coef.lower() == "random"
        is_fixed  = isinstance(coef, (list, tuple))

        if (is_random + is_fixed) != 1:
            raise ValueError(f"Head {i}: 'forcing.coef' must be a list OR the string 'random'.")

        if is_random:
            m = len(freq)
            scale = 0.0 if m == 0 else 2.0 / m
            coefs = list(rng.uniform(0.0, scale, size=m))
        else:
            coefs = list(coef)
            if len(coefs) != len(freq):
                raise ValueError(
                    f"Head {i}: len(coef)={len(coefs)} must equal len(freq)={len(freq)}."
                )
            
        mu_list.append(mu)
        w_list.append(w0)
        zeta_list.append(zeta)
        ic_python_list.append([float(ic[0]), float(ic[1])])
        forcing_freq.append(freq)
        forcing_coef.append(coefs)

    t = cfg.get("train")

    return {
        "experiment": cfg.get("experiment"),
        "N": int(cfg["N"]),
        "t_max": cfg["t_max"],
        "k": k,
        "seed": cfg.get("seed"),
        "w_list": w_list,
        "zeta_list": zeta_list,
        "mu_list": mu_list,
        "forcing_freq": forcing_freq,
        "forcing_coef": forcing_coef,
        "ic_python_list": ic_python_list,
        "lr": float(t.get("lr")),
        "num_iter": t.get("num_iter"),
        "step_size": t.get("step_size"),
        "gamma": t.get("gamma"),
        "w_ode": t.get("w_ode"),
        "w_ic": t.get("w_ic"),
        "method": t.get("method"),
        "hidden_layers": t.get("hidden_layers"),
        "use_sine": t.get("use_sine"),
        "use_fourier": t.get("use_fourier"),
        "n_frequencies": t.get("n_frequencies"),
        "scale": t.get("scale"),
        "bias": t.get("bias"),
    }
