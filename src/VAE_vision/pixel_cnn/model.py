import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    """Conv2d with a causal mask enforcing raster-order autoregression.

    Type A: excludes the current position (first layer only — input is raw code
            indices, so seeing the current value would leak the answer).
    Type B: includes the current position (all later layers — inputs are
            intermediate features with no raw index information).
    """
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        h, w = self.weight.shape[2], self.weight.shape[3]
        mask = torch.zeros_like(self.weight)
        mask[:, :, :h // 2, :] = 1                        # all rows above center
        mask[:, :, h // 2, :w // 2] = 1                   # left of center on same row
        if mask_type == "B":
            mask[:, :, h // 2, w // 2] = 1                # center itself (Type B only)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _GatedActivation(nn.Module):
    """tanh(a) * sigmoid(b) where a and b are the two halves of the channel dim."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _ChannelLayerNorm(channels)
        self.conv = MaskedConv2d("B", channels, channels * 2, kernel_size=3, padding=1)
        self.gate = _GatedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate(self.conv(self.norm(x)))


class PixelCNN(nn.Module):
    def __init__(
        self,
        num_codes: int = 512,
        embed_dim: int = 192,
        n_layers: int = 6,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, embed_dim)

        self.input_conv = MaskedConv2d("A", embed_dim, embed_dim * 2, kernel_size=3, padding=1)
        self.input_gate = _GatedActivation()

        self.layers = nn.ModuleList([_ResidualBlock(embed_dim) for _ in range(n_layers)])

        self.head = nn.Sequential(
            _ChannelLayerNorm(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, num_codes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, H, W) int64 code indices
        returns: (B, num_codes, H, W) logits
        """
        x = self.embedding(x).permute(0, 3, 1, 2)    # (B, D, H, W)
        x = self.input_gate(self.input_conv(x))        # (B, D, H, W)
        for block in self.layers:
            x = block(x)
        return self.head(x)                            # (B, num_codes, H, W)
