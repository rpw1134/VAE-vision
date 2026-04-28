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
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(512, 64)
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)

    def forward(self, x):
        # x is (B, 64, 16, 16)
        # we want to map each 64-dim vector to the nearest codebook entry
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        weights = self.codebook.weight  # (512, 64)
        distances = torch.cdist(x_flat, weights)  # takes pairwise distances L2 (B*H*W, 512)
        indices = torch.argmin(distances, dim=1)  # (B*H*W,) for each index
        quantized = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        commitment_loss = torch.mean((x - quantized.detach()) ** 2)
        codebook_loss = torch.mean((x.detach() - quantized) ** 2)
        unique_codes = indices.unique().numel()

        return x + (quantized - x).detach(), commitment_loss, codebook_loss, unique_codes

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
        x, commitment, codebook, unique_codes = self.quantizer(x)
        x = self.decoder(x)
        return x, commitment, codebook, unique_codes
