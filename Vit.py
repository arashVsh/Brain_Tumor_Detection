import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=64, in_channels=1, num_classes=2, hidden_dim=64, num_layers=6, num_heads=8, mlp_dim=128):
        super(VisionTransformer, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=mlp_dim), num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, 'Input tensor shape must be square for Vision Transformer'
        assert H % self.patch_size == 0, 'Image dimensions must be divisible by patch size'
        P = H // self.patch_size
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.positional_embedding.repeat(B, 1, 1), x), dim=1)
        x = self.transformer_encoder(x)
        x = x.mean(1)  # Global average pooling
        x = self.fc(x)
        return x