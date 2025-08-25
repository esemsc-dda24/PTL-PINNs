import torch
import torch.nn as nn
import numpy as np
import json

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    
class FourierFeatures(nn.Module):
    def __init__(self, n_frequencies, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(scale * torch.randn(n_frequencies, 1), requires_grad=False)  # [F, 1]

    def forward(self, x):  # x shape: [N, 1]
        x_proj = 2 * torch.pi * x @ self.B.T  # [N, F]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [N, 2F]

class Multihead_model_fourier(nn.Module):
    def __init__(self, k, hidden_act=[torch.sin, nn.Tanh(), nn.Tanh()], use_sine=False, 
                use_fourier=False, omega_0 = 1, n_frequencies=16,
                scale = 1.0, bias=False, HIDDEN_LAYERS=[128, 128, 256]):
        super().__init__()
        self.use_sine = use_sine
        self.use_fourier = use_fourier
        self.HIDDEN_LAYERS = HIDDEN_LAYERS

        input_dim = 1
        if use_fourier:
            self.fourier = FourierFeatures(n_frequencies=n_frequencies, scale=scale)
            input_dim = 2 * n_frequencies

        if use_sine:
            self.linear1 = SineLayer(input_dim, HIDDEN_LAYERS[0], omega_0=omega_0)
            self.linear2 = SineLayer(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
            self.linear3 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
        else:
            self.linear1 = nn.Linear(input_dim, HIDDEN_LAYERS[0])
            self.linear2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
            self.linear3 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
            self.act = hidden_act

        self.final_layers = nn.ModuleList([nn.Linear(HIDDEN_LAYERS[1], 1, bias=bias) for _ in range(k)])
        self.k = k
        self.double()

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)

        if self.use_sine:
            out = self.linear1(x)
            out = self.linear2(out)
            out = self.linear3(out)
        else:
            out = self.act[0](self.linear1(x))
            out = self.act[1](self.linear2(out))
            out = self.act[2](self.linear3(out))

        out1 = out[:, :self.HIDDEN_LAYERS[1]]
        out2 = out[:, self.HIDDEN_LAYERS[1]:]
        output = []
        for i in range(self.k):
            first = self.final_layers[i](out1)
            second = self.final_layers[i](out2)
            concat = torch.cat((first, second), axis=1)
            output.append(concat)
        return torch.stack(output), out


def load_model(path, name):

    with open(f'{path}/training_log.json', 'r') as f:
        training_log = json.load(f)

    k = training_log['k']
    bias = training_log['bias']
    use_sine = training_log['use_sine']
    use_fourier = training_log['use_fourier']
    scale = training_log['scale']
    n_frequencies = training_log['n_frequencies']
    hidden_layers = training_log['hidden_layers']

    if use_fourier:
        print(k, bias, use_sine, use_fourier, scale, n_frequencies, hidden_layers)
        pinn = Multihead_model_fourier(k=k, bias=bias, use_sine=use_sine, use_fourier=use_fourier, scale=scale, n_frequencies=n_frequencies, HIDDEN_LAYERS=hidden_layers)
    else:
        print(k, bias, use_sine, use_fourier, hidden_layers)
        pinn = Multihead_model_fourier(k=k, bias=bias, HIDDEN_LAYERS=hidden_layers)

    pinn.load_state_dict(torch.load(f'{path}/{name}'))

    return pinn, training_log