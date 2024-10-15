import torch
import torch.nn.functional as F

class AddNoiseProcess:
    """
    A class to represent the process of adding noise to an image over a series of timesteps.

    Attributes
    ----------
    T : int
        Number of timesteps for the noise addition process.
    betas : torch.Tensor
        Linear schedule of beta values over the timesteps.
    alphas : torch.Tensor
        Values representing 1 - betas.
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas over the timesteps.
    alphas_cumprod_prev : torch.Tensor
        Cumulative product of alphas for the previous timestep.
    sqrt_recip_alphas : torch.Tensor
        Square root of the reciprocal of alphas.
    sqrt_alphas_cumprod : torch.Tensor
        Square root of the cumulative product of alphas.
    sqrt_one_minus_alphas_cumprod : torch.Tensor
        Square root of 1 minus the cumulative product of alphas.
    posterior_variance : torch.Tensor
        Posterior variance calculated using betas and alphas_cumprod.

    """
    def __init__(self, T=300):
        self.T = T
        self.betas = self.linear_beta_schedule(timesteps=T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
