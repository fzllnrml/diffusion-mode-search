import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .noise_schedule import NoiseSchedule
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)

class EpsilonNet(nn.Module):
    _activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }

    def __init__(self, input_dim, hidden_dims, activation="silu", T=1000):
        super().__init__()
        self.T = T

        act_cls = self._activations.get(activation, nn.SiLU)

        layers = []
        prev_dim = input_dim + 1
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), act_cls()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t):
        t_norm = t.float() / (self.T - 1)
        t_emb = t_norm.unsqueeze(-1)
        inp = torch.cat([x_t, t_emb], dim=-1)
        return self.net(inp)


class NFECounter:
    def __init__(self, model):
        self.model = model
        self.nfe = 0

    @torch.no_grad()
    def __call__(self, x, t):
        self.nfe += x.shape[0]
        return self.model(x, t)

    def reset(self):
        self.nfe = 0


class DiffusionModel:
    def __init__(
        self,
        dim=1,
        T=1000,
        beta_start=1e-4,
        beta_end=0.02,
        hidden_dims=None,
        activation="silu",
        device=None,
    ):
        self.dim = dim

        self.device = device or resolve_device("auto")


        self.schedule = NoiseSchedule.linear(T, beta_start, beta_end, self.device)

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.net = EpsilonNet(dim, hidden_dims, activation, T).to(self.device)

        self._nfe_counter = None

        self._train_step = 0
        self._last_loss = float("inf")

        logger.info(
            "DiffusionModel: dim=%d, T=%d, device=%s, params=%d",
            dim, T, self.device,
            sum(p.numel() for p in self.net.parameters()),
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        s = self.schedule
        sqrt_alpha = s.extract(s.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = s.extract(s.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    @torch.no_grad()
    def score(self, x, t):
        self.net.eval()
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

        if self._nfe_counter is not None:
            eps_pred = self._nfe_counter(x, t_tensor)
        else:
            eps_pred = self.net(x, t_tensor)

        sigma_t = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        score_val = -eps_pred / sigma_t

        return score_val.squeeze(0) if single else score_val

    def score_numpy(self, x, t):
        x_t = torch.tensor(x, device=self.device, dtype=torch.float32)
        if x_t.dim() == 0:
            x_t = x_t.unsqueeze(0)
        if x_t.dim() == 1 and self.dim == 1:
            x_t = x_t.unsqueeze(-1)
        return self.score(x_t, t).cpu().numpy()

    def enable_nfe_counting(self):
        self._nfe_counter = NFECounter(self.net)

    def reset_nfe(self):
        if self._nfe_counter:
            self._nfe_counter.reset()

    @property
    def nfe(self):
        return self._nfe_counter.nfe if self._nfe_counter else 0

    def train_on_data(
        self,
        sample_fn,
        num_steps=200_000,
        batch_size=512,
        lr=1e-3,
        lr_min=1e-5,
        scheduler="cosine",
        log_every=1000,
        save_every=0,
        save_path=None,
    ):
        self.net.train()

        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        sched = None
        if scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_steps, eta_min=lr_min
            )

        loss_fn = nn.MSELoss()
        loss_history = []
        T = self.schedule.T

        logger.info("Начало обучения: %d шагов, batch=%d, lr=%.1e", num_steps, batch_size, lr)

        for step in range(self._train_step + 1, self._train_step + num_steps + 1):
            x0 = sample_fn(batch_size)
            t = torch.randint(0, T, (batch_size,), device=self.device)
            eps = torch.randn_like(x0)

            x_t = self.q_sample(x0, t, noise=eps)
            eps_pred = self.net(x_t, t)
            loss = loss_fn(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()

            self._train_step = step
            self._last_loss = loss.item()

            if log_every and step % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Шаг %d/%d  Loss: %.5f  LR: %.2e",
                    step, self._train_step, self._last_loss, current_lr,
                )
                loss_history.append(self._last_loss)

            if save_every and save_path and step % save_every == 0:
                self.save_checkpoint(save_path)

        logger.info("Обучение завершено. loss: %.5f", self._last_loss)
        return loss_history

    @torch.no_grad()
    def sample(self, n=1000):
        self.net.eval()
        s = self.schedule

        x = torch.randn(n, self.dim, device=self.device)

        for t_step in reversed(range(s.T)):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)

            eps_pred = self.net(x, t)

            beta_t = s.extract(s.betas, t, x.shape)
            sqrt_omc = s.extract(s.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip = s.extract(s.sqrt_recip_alphas, t, x.shape)

            mean = sqrt_recip * (x - beta_t / sqrt_omc * eps_pred)

            if t_step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean

        result = x.cpu().numpy()
        return result.squeeze(-1) if self.dim == 1 else result

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.net.state_dict(),
            "train_step": self._train_step,
            "last_loss": self._last_loss,
            "dim": self.dim,
            "T": self.schedule.T,
            "config": {
                "hidden_dims": [
                    m.out_features
                    for m in self.net.net
                    if isinstance(m, nn.Linear)
                ][:-1],
            },
        }

        torch.save(checkpoint, path)
        logger.info(
            "Чекпоинт сохранён: %s (step=%d, loss=%.5f)",
            path, self._train_step, self._last_loss,
        )

    def load_checkpoint(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["state_dict"])
        self._train_step = checkpoint.get("train_step", 0)
        self._last_loss = checkpoint.get("last_loss", float("inf"))

        logger.info(
            "Чекпоинт загружен: %s (step=%d, loss=%.5f)",
            path, self._train_step, self._last_loss,
        )
        return checkpoint

    @classmethod
    def from_checkpoint(cls, path, device=None):
        checkpoint = torch.load(
            path, map_location=device or "cpu", weights_only=False,
        )

        model = cls(
            dim=checkpoint.get("dim", 1),
            T=checkpoint.get("T", 1000),
            hidden_dims=checkpoint.get("config", {}).get("hidden_dims", [256, 256, 256]),
            device=device,
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        model._train_step = checkpoint.get("train_step", 0)
        model._last_loss = checkpoint.get("last_loss", float("inf"))
        return model
