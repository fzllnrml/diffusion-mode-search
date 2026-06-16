import torch
from dataclasses import dataclass

@dataclass
class NoiseSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor

    @staticmethod
    def linear(T, beta_start, beta_end, device):
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]]
        )

        return NoiseSchedule(
            T=T,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        )


    def sigma(self, t):
        return float(self.sqrt_one_minus_alphas_cumprod[t].item())


    def extract(self, a, t, x_shape):
        out = a.gather(-1, t.view(-1))
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out
