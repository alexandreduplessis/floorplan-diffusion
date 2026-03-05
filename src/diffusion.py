import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion:
    """
    DDPM forward and reverse diffusion process.

    This class manages the noise schedule and provides methods for:
    - Adding noise (forward process)
    - Computing training loss
    - Sampling (reverse process)
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # For posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """Extract values from a at indices t, reshape for broadcasting with x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: add noise to x0 at timestep t.

        Args:
            x0: (B, C, H, W) clean latent
            t: (B,) integer timesteps
            noise: optional pre-sampled noise
        Returns:
            xt: (B, C, H, W) noisy latent at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape).to(x0.device)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape).to(x0.device)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def p_losses(self, denoiser, x0, condition_latent, t):
        """
        Compute training loss.

        Args:
            denoiser: the ViT model that takes (concat, t) and predicts noise
            x0: (B, 4, 64, 64) clean floor plan latent
            condition_latent: (B, 4, 64, 64) condition encoding
            t: (B,) timesteps
        Returns:
            loss: scalar MSE loss
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        # Concatenate noisy latent with condition
        model_input = torch.cat([xt, condition_latent], dim=1)  # (B, 8, 64, 64)

        # Predict noise
        predicted_noise = denoiser(model_input, t)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, denoiser, xt, condition_latent, t):
        """
        Single reverse diffusion step: sample x_{t-1} from p_theta(x_{t-1}|x_t).

        Args:
            denoiser: the ViT model
            xt: (B, C, H, W) noisy latent at timestep t
            condition_latent: (B, 4, 64, 64) condition encoding
            t: (B,) current timestep (same value for all batch elements)
        Returns:
            x_{t-1}: (B, C, H, W)
        """
        model_input = torch.cat([xt, condition_latent], dim=1)
        predicted_noise = denoiser(model_input, t)

        beta_t = self._extract(self.betas, t, xt.shape).to(xt.device)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, xt.shape).to(xt.device)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape).to(xt.device)

        # Predicted mean
        mean = sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise)

        if t[0] == 0:
            return mean

        posterior_var = self._extract(self.posterior_variance, t, xt.shape).to(xt.device)
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(self, denoiser, condition_latent, shape, device='cuda'):
        """
        Full reverse process: generate samples from noise.

        Args:
            denoiser: the ViT model
            condition_latent: (B, 4, 64, 64) condition encoding
            shape: tuple (B, C, H, W) for the output shape
            device: torch device
        Returns:
            x0: (B, C, H, W) generated clean latent
        """
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(denoiser, x, condition_latent, t)

        return x

    @torch.no_grad()
    def ddim_sample(self, denoiser, condition_latent, shape, device='cuda',
                     num_inference_steps=50, eta=0.0):
        """
        DDIM sampling for faster inference.

        Args:
            denoiser: the ViT model
            condition_latent: (B, 4, 64, 64)
            shape: (B, C, H, W)
            device: torch device
            num_inference_steps: number of denoising steps (default 50, much faster than 1000)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
        Returns:
            x0: (B, C, H, W) generated clean latent
        """
        # Create sub-sequence of timesteps
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(shape, device=device)

        for i, t_curr in enumerate(timesteps):
            t = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)

            model_input = torch.cat([x, condition_latent], dim=1)
            predicted_noise = denoiser(model_input, t)

            alpha_bar_t = self.alphas_cumprod[t_curr].to(device)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev].to(device)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Predicted x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # clip for stability

            # Direction pointing to xt
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * predicted_noise

            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

            if sigma_t > 0:
                x = x + sigma_t * torch.randn_like(x)

        return x
