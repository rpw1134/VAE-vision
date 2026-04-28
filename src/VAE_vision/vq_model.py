import torch
import torch.nn.functional as F
from torch import nn


# need an encode 128 x 128 x 3 (input, RGB) to map the image down to a space of 16 x 16 x 64 (arbitrary S, D dim latent)
# then NN component to hold to codebook which should be some 512 x 64
# then a decoder to take the 16 x 16 x 64 and reconstruct back to 128 x 128 x 3

class VQEncoder(nn.Module):
    def __init__(self):
        super(VQEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),   # (B, 256, 64, 64)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # (B, 512, 32, 32)
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 16, 16)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, decay: float = 0.99):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.decay = decay

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, 0.0, 1.0)

        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone())

    def initialize_from_data(self, z: torch.Tensor) -> None:
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C).detach()
        idx = torch.randint(0, z_flat.shape[0], (self.num_embeddings,), device=z.device)
        self.codebook.weight.data = z_flat[idx]
        self.ema_weight = z_flat[idx].clone()
        self.ema_cluster_size = torch.ones(self.num_embeddings, device=z.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)

        # ||x - e||² = ||x||² + ||e||² - 2·x·eᵀ
        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)
        distances = x_sq + e_sq - 2 * (x_flat @ self.codebook.weight.T)  # (B*H*W, K)

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        if self.training:
            one_hot = F.one_hot(indices, self.num_embeddings).float()  # (B*H*W, K)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * one_hot.sum(0)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * (one_hot.T @ x_flat)
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.codebook.weight.data = self.ema_weight / smoothed.unsqueeze(1)

            # revive dead codes by splitting the dominant code: copy it then add noise
            dead = self.ema_cluster_size < 1.0
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                top_idx = self.ema_cluster_size.argmax()
                top_vec = self.codebook.weight.data[top_idx]
                new_codes = top_vec + 0.1 * torch.randn(n_dead, C, device=x.device)
                self.codebook.weight.data[dead] = new_codes
                self.ema_weight[dead] = new_codes
                self.ema_cluster_size[dead] = 1.0

        commitment_loss = torch.mean((x - quantized.detach()) ** 2)
        unique_codes = torch.tensor(indices.unique().numel(), dtype=torch.float32, device=x.device)

        return x + (quantized - x).detach(), commitment_loss, unique_codes

class VQDecoder(nn.Module):
    def __init__(self):
        super(VQDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 512, kernel_size=4, stride=2, padding=1),  # (B, 512, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # (B, 256, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),   # (B, 3, 128, 128)
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class VQModel(nn.Module):
    def __init__(self):
        super(VQModel, self).__init__()
        self.encoder = VQEncoder()
        self.quantizer = VectorQuantizer()
        self.decoder = VQDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x, commitment, unique_codes = self.quantizer(x)
        x = self.decoder(x)
        return x, commitment, unique_codes
