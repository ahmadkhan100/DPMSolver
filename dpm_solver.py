"""
DPM-Solver++: Fast ODE Solver for Diffusion Probabilistic Models
Research Implementation

Based on:
- Lu et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling" (NeurIPS 2022)
- Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models" (arXiv 2022)
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
from typing import Optional, Callable, Union


def get_time_steps(skip_type: str, t_T: float, t_0: float, N: int, device: torch.device) -> torch.Tensor:
    """Generate time steps for sampling."""
    if skip_type == 'logSNR':
        lambda_T = math.log(1 / (1 - 1e-4))
        lambda_0 = math.log(1 / (1 - 1))
        logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1, device=device)
        return logSNR_steps
    elif skip_type == 'time_uniform':
        return torch.linspace(t_T, t_0, N + 1, device=device)
    elif skip_type == 'time_quadratic':
        t_order = 2
        t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1, device=device)**t_order
        return t


def noise_schedule_marginal_log_mean_coeff(t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0) -> torch.Tensor:
    """Marginal log mean coefficient for VP SDE."""
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    return log_mean_coeff


def noise_schedule_marginal_std(t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0) -> torch.Tensor:
    """Marginal standard deviation for VP SDE."""
    log_mean_coeff = noise_schedule_marginal_log_mean_coeff(t, beta_0, beta_1)
    log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
    return torch.exp(log_std)


def noise_schedule_marginal_lambda(t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0) -> torch.Tensor:
    """Marginal lambda for VP SDE: lambda_t = log(alpha_t / sigma_t)."""
    log_mean_coeff = noise_schedule_marginal_log_mean_coeff(t, beta_0, beta_1)
    log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
    return log_mean_coeff - log_std


def model_wrapper(model: torch.nn.Module, noise_schedule: str = "VP", 
                 model_type: str = "noise", guidance_type: str = "uncond", guidance_scale: float = 1.0):
    """Wrapper for different model types and guidance."""
    
    def get_model_input_time(t_continuous):
        return (t_continuous - 1. / 1000) * 1000

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        
        if guidance_type == "uncond":
            return model(x, t_input)
        elif guidance_type == "classifier":
            return model(x, t_input, cond)
        elif guidance_type == "classifier-free":
            if cond is None:
                return model(x, t_input)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_input] * 2)
                c_in = torch.cat([torch.zeros_like(cond), cond])
                noise_uncond, noise_cond = model(x_in, t_in, c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    def cond_fn(x, t, cond=None):
        if model_type == "noise":
            return noise_pred_fn(x, t, cond)
        elif model_type == "x_start":
            noise = noise_pred_fn(x, t, cond)
            alpha_t, sigma_t = noise_schedule_marginal_log_mean_coeff(t).exp(), noise_schedule_marginal_std(t)
            return (x - alpha_t * noise) / sigma_t

    return cond_fn


class DPMSolver:
    """DPM-Solver for fast sampling of diffusion models."""

    def __init__(self, model_fn: Callable, noise_schedule: str = "VP", 
                 algorithm_type: str = "dpmsolver++", correcting_x0_fn: str = "none"):
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.algorithm_type = algorithm_type
        self.correcting_x0_fn = correcting_x0_fn

    def get_time_steps(self, skip_type: str, t_T: Union[float, torch.Tensor], 
                      t_0: Union[float, torch.Tensor], N: int, device: torch.device) -> torch.Tensor:
        return get_time_steps(skip_type, t_T, t_0, N, device)

    def denoise_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert any prediction to the noise prediction for DPM-Solver."""
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.model(x, t)

    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return x_0 prediction from x_t."""
        noise = self.model(x, t)
        alpha_t = noise_schedule_marginal_log_mean_coeff(t).exp()
        sigma_t = noise_schedule_marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        
        if self.correcting_x0_fn == "dynamic_thresholding":
            x0 = self.dynamic_thresholding_fn(x0)
        
        return x0

    def dynamic_thresholding_fn(self, x0: torch.Tensor, clamp_value: float = 1.0) -> torch.Tensor:
        """Dynamic thresholding for x0 prediction."""
        dims = tuple(range(1, x0.ndim))
        abs_x0 = torch.abs(x0)
        s = torch.quantile(abs_x0.view(x0.shape[0], -1), 0.995, dim=1, keepdim=True)
        s = torch.maximum(s, torch.ones_like(s) * clamp_value)[(...,) + (None,) * len(dims)]
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal lambda function."""
        return noise_schedule_marginal_lambda(t)

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal alpha function."""
        return noise_schedule_marginal_log_mean_coeff(t).exp()

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal standard deviation function."""
        return noise_schedule_marginal_std(t)

    def dpm_solver_first_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                               model_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """First order update for DPM-Solver."""
        if model_s is None:
            model_s = self.denoise_fn(x, s)
        
        lambda_s, lambda_t = self.marginal_lambda(s), self.marginal_lambda(t)
        h = lambda_t - lambda_s
        
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(h)
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x + \
                  self.marginal_alpha(t) * phi_1 * model_s
        else:
            phi_1 = torch.expm1(h)
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x - \
                  self.marginal_std(t) * phi_1 * model_s
        
        return x_t

    def dpm_solver_second_update(self, x: torch.Tensor, s_list: list, t: torch.Tensor,
                                model_s_list: list, solver_type: str = "dpmsolver") -> torch.Tensor:
        """Second order update for DPM-Solver."""
        s0, s1 = s_list[-1], s_list[-2]
        m0, m1 = model_s_list[-1], model_s_list[-2]
        
        lambda_s0, lambda_s1, lambda_t = self.marginal_lambda(s0), self.marginal_lambda(s1), self.marginal_lambda(t)
        h_0 = lambda_s0 - lambda_s1
        h_1 = lambda_t - lambda_s0
        r0 = h_0 / h_1
        
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(h_1)
            phi_2 = phi_1 / h_1 - 1.
            
            if solver_type == "dpmsolver":
                x_t = self.marginal_alpha(t) / self.marginal_alpha(s0) * x + \
                      self.marginal_alpha(t) * phi_1 * m0 + \
                      self.marginal_alpha(t) * phi_2 * (m0 - m1) / r0
            elif solver_type == "taylor":
                x_t = self.marginal_alpha(t) / self.marginal_alpha(s0) * x + \
                      self.marginal_alpha(t) * phi_1 * m0 + \
                      self.marginal_alpha(t) * ((phi_2 / r0) * (m0 - m1))
        else:
            phi_1 = torch.expm1(h_1)
            phi_2 = phi_1 / h_1 - 1.
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s0) * x - \
                  self.marginal_std(t) * phi_1 * m0 - \
                  self.marginal_std(t) * phi_2 * (m0 - m1) / r0
        
        return x_t

    def dpm_solver_third_update(self, x: torch.Tensor, s_list: list, t: torch.Tensor,
                               model_s_list: list) -> torch.Tensor:
        """Third order update for DPM-Solver."""
        s0, s1, s2 = s_list[-1], s_list[-2], s_list[-3]
        m0, m1, m2 = model_s_list[-1], model_s_list[-2], model_s_list[-3]
        
        lambda_s0, lambda_s1, lambda_s2, lambda_t = self.marginal_lambda(s0), self.marginal_lambda(s1), \
                                                   self.marginal_lambda(s2), self.marginal_lambda(t)
        h_0 = lambda_s0 - lambda_s1
        h_1 = lambda_s1 - lambda_s2
        h_2 = lambda_t - lambda_s0
        r0, r1 = h_0 / h_2, h_1 / h_2
        
        phi_1 = torch.expm1(h_2)
        phi_2 = phi_1 / h_2 - 1.
        phi_3 = phi_2 / h_2 - 0.5
        
        D1_0 = (m0 - m1) / r0
        D1_1 = (m1 - m2) / r1
        D2 = (D1_0 - D1_1) / (r0 + r1)
        
        if self.algorithm_type == "dpmsolver++":
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s0) * x + \
                  self.marginal_alpha(t) * (phi_1 * m0 + phi_2 * D1_0 + phi_3 * D2)
        else:
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s0) * x - \
                  self.marginal_std(t) * (phi_1 * m0 + phi_2 * D1_0 + phi_3 * D2)
        
        return x_t

    def singlestep_dpm_solver_second_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                                          use_corrector: bool = True, model_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Singlestep second order update."""
        if model_s is None:
            model_s = self.denoise_fn(x, s)
        
        lambda_s, lambda_t = self.marginal_lambda(s), self.marginal_lambda(t)
        h = lambda_t - lambda_s
        
        # Midpoint method
        s_mid = torch.exp(0.5 * (torch.log(s) + torch.log(t)))
        x_mid = self.dpm_solver_first_update(x, s, s_mid, model_s=model_s)
        model_mid = self.denoise_fn(x_mid, s_mid)
        
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(h)
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x + \
                  self.marginal_alpha(t) * phi_1 * model_mid
        else:
            phi_1 = torch.expm1(h)
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x - \
                  self.marginal_std(t) * phi_1 * model_mid
        
        return x_t

    def singlestep_dpm_solver_third_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                                         use_corrector: bool = True, model_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Singlestep third order update."""
        if model_s is None:
            model_s = self.denoise_fn(x, s)
        
        lambda_s, lambda_t = self.marginal_lambda(s), self.marginal_lambda(t)
        h = lambda_t - lambda_s
        
        # Third order requires two intermediate points
        s_1 = torch.exp(torch.log(s) + h / 3.)
        s_2 = torch.exp(torch.log(s) + 2. * h / 3.)
        
        x_1 = self.dpm_solver_first_update(x, s, s_1, model_s=model_s)
        model_1 = self.denoise_fn(x_1, s_1)
        
        x_2 = self.dpm_solver_first_update(x, s, s_2, model_s=model_s)
        model_2 = self.denoise_fn(x_2, s_2)
        
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x + \
                  self.marginal_alpha(t) * (phi_1 * model_s + phi_2 * (model_1 - model_s) * 3. + \
                                          phi_3 * (model_2 - 2. * model_1 + model_s) * 9.)
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            
            x_t = self.marginal_alpha(t) / self.marginal_alpha(s) * x - \
                  self.marginal_std(t) * (phi_1 * model_s + phi_2 * (model_1 - model_s) * 3. + \
                                        phi_3 * (model_2 - 2. * model_1 + model_s) * 9.)
        
        return x_t

    def singlestep_dpm_solver_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                                   order: int, use_corrector: bool = True,
                                   x_prev: Optional[torch.Tensor] = None,
                                   model_s: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Singlestep DPM-Solver update."""
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, model_s=model_s)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, use_corrector, model_s=model_s)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, use_corrector, model_s=model_s)

    def multistep_dpm_solver_update(self, x: torch.Tensor, model_prev_list: list,
                                  t_prev_list: list, t: torch.Tensor, order: int,
                                  solver_type: str = "dpmsolver") -> torch.Tensor:
        """Multistep DPM-Solver update."""
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.dpm_solver_second_update(x, t_prev_list, t, model_prev_list, solver_type=solver_type)
        elif order == 3:
            return self.dpm_solver_third_update(x, t_prev_list, t, model_prev_list)

    def sample_adaptive(self, x: torch.Tensor, t_T: float, t_0: float, order: int, 
                       atol: float, rtol: float, method: str) -> torch.Tensor:
        """Adaptive sampling with error control."""
        t = t_T
        x_t = x.clone()
        
        while t > t_0:
            # Estimate optimal step size
            h_init = min(0.05, t - t_0)
            
            # Compute solution with step h and h/2
            x_h = self.singlestep_dpm_solver_update(x_t, torch.tensor(t), torch.tensor(t - h_init), order)
            
            x_h2_1 = self.singlestep_dpm_solver_update(x_t, torch.tensor(t), torch.tensor(t - h_init/2), order)
            x_h2_2 = self.singlestep_dpm_solver_update(x_h2_1, torch.tensor(t - h_init/2), torch.tensor(t - h_init), order)
            
            # Error estimation
            error = torch.norm(x_h - x_h2_2) / torch.norm(x_h2_2)
            
            # Accept or reject step
            if error < atol + rtol * torch.norm(x_h2_2):
                x_t = x_h2_2
                t = t - h_init
            
            # Update step size
            h_init = h_init * min(2., max(0.5, 0.9 * (atol / error) ** (1. / (order + 1))))
        
        return x_t

    @autocast()
    def sample(self, x: torch.Tensor, steps: int = 20, t_start: Optional[float] = None,
              t_end: Optional[float] = None, order: int = 3, skip_type: str = 'time_uniform',
              method: str = 'singlestep', lower_order_final: bool = True,
              denoise_to_zero: bool = False, solver_type: str = 'dpmsolver',
              atol: float = 0.0078, rtol: float = 0.05) -> torch.Tensor:
        """
        Sample from diffusion models with DPM-Solver.
        
        Args:
            x: Initial noise
            steps: Number of sampling steps
            t_start: Start time (default: 1.0)
            t_end: End time (default: 1e-3)
            order: Solver order (1, 2, or 3)
            skip_type: Time step schedule
            method: 'singlestep', 'multistep', or 'adaptive'
            lower_order_final: Use lower order for final steps
            denoise_to_zero: Denoise to t=0
            solver_type: 'dpmsolver' or 'taylor'
            atol: Absolute tolerance for adaptive method
            rtol: Relative tolerance for adaptive method
        """
        device = x.device
        t_0 = 1. / 1000 if t_end is None else t_end
        t_T = 1. if t_start is None else t_start

        if method == 'adaptive':
            return self.sample_adaptive(x, t_T, t_0, order, atol, rtol, method)
        
        timesteps = self.get_time_steps(skip_type, t_T, t_0, steps, device)
        
        if method == 'multistep':
            return self.sample_multistep(x, timesteps, order, solver_type)
        else:
            return self.sample_singlestep(x, timesteps, order, steps)

    def sample_multistep(self, x: torch.Tensor, timesteps: torch.Tensor, 
                        order: int, solver_type: str) -> torch.Tensor:
        """Multistep sampling."""
        model_prev_list = []
        t_prev_list = []
        
        for i, t in enumerate(timesteps):
            t_prev_list.append(t)
            model_prev_list.append(self.denoise_fn(x, t))
            
            if len(t_prev_list) > order:
                t_prev_list.pop(0)
                model_prev_list.pop(0)
            
            if i < len(timesteps) - 1:
                order_used = min(len(model_prev_list), order)
                x = self.multistep_dpm_solver_update(
                    x, model_prev_list, t_prev_list, timesteps[i + 1], order_used, solver_type
                )
        
        return x

    def sample_singlestep(self, x: torch.Tensor, timesteps: torch.Tensor, 
                         order: int, steps: int) -> torch.Tensor:
        """Singlestep sampling."""
        for i in range(len(timesteps) - 1):
            t_i, t_ip1 = timesteps[i], timesteps[i + 1]
            x = self.singlestep_dpm_solver_update(x, t_i, t_ip1, order)
        
        return x 