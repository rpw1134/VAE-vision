import torch
from torch import nn

# need an encode 128 x 128 x 3 (input, RGB) to map the image down to a space of 16 x 16 x 64 (arbitrary S, D dim latent)
# then NN component to hold to codebook which should be some 512 x 64
# then a decoder to take the 16 x 16 x 64 and reconstruct back to 128 x 128 x 3

class VQEncoder(nn.Module):
    def __init__(self):
        super(VQEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 16, 16)
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Embedding(512, 64)

    def forward(self, x):
        code = self.codebook(x)
        return code

class VQDecoder(nn.Module):
    def __init__(self):
        super(VQDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(), # normalize pixels to between 0 and 1
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
        x = self.quantizer(x)
        x = self.decoder(x)
        return x
