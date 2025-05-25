# DPM-Solver++: Fast ODE Solver for Diffusion Models

Research implementation of DPM-Solver++ for fast sampling from diffusion probabilistic models.

## Papers

- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling](https://arxiv.org/abs/2206.00927) (NeurIPS 2022)
- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) (2022)

## Usage

```python
import torch
from dpm_solver import DPMSolver, model_wrapper

# Wrap your diffusion model
model_fn = model_wrapper(model, guidance_type="classifier-free", guidance_scale=7.5)

# Initialize solver
solver = DPMSolver(model_fn, algorithm_type="dpmsolver++")

# Sample
x_T = torch.randn(1, 3, 512, 512).cuda()
samples = solver.sample(x_T, steps=20, order=3, method='singlestep')
```

## Mathematical Formulation

DPM-Solver++ converts the diffusion ODE:
```
dx/dt = (f(t)x + g(t)²∇_x log p_t(x)) / 2
```

Into log-SNR coordinates λ_t = log(α_t/σ_t) where the ODE becomes:
```
dx/dλ = (e^(-λ)σ_λ/α_λ)(α_λx_0_θ(x,λ) - x)
```

The solver uses exponential integrators with order 1, 2, or 3 for fast convergence.

## Installation

```bash
pip install torch numpy
```

## Methods

- **Singlestep**: High-order solver with intermediate evaluations
- **Multistep**: Adams-Bashforth style multistep method  
- **Adaptive**: Automatic step size control with error estimation

## Algorithms

- `dpmsolver++`: Data prediction variant (recommended)
- `dpmsolver`: Noise prediction variant 
