import torch
import torch.nn as nn

LATENT_DIM = 128
ENCODER_CHANNELS = [32, 64, 128, 256]  # spatial: 128 -> 64 -> 32 -> 16 -> 8


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# (B, 256, 8, 8)
            nn.ReLU(),
        )
        self.mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.log_var = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.mu(x), self.log_var(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # (B, 3, 128, 128)
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.linear(z)
        x = x.view(-1, 256, 8, 8)
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
