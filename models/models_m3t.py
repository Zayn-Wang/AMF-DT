# models_m3t.py
"""
Model definitions for Sub-network 5 (M3T-based survival model).

Contains:
- CNN3DBlock
- MultiPlane_MultiSlice_Extract_Project
- EmbeddingLayer
- MultiHeadAttention / TransformerEncoder
- M3T backbone
- M3t survival head: outputs OS, PFS, Age, 9D clinical multi-label logits
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat


# -------------------------------------------------------------------------
# 3D CNN block
# -------------------------------------------------------------------------
class CNN3DBlock(nn.Module):
    """
    3D CNN block to extract volumetric features from MRI.

    Input:  I ∈ R^{B×C×L×W×H}
    Output: X ∈ R^{B×C3d×L×W×H}
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


# -------------------------------------------------------------------------
# Multi-plane / multi-slice extraction + 2D GAP + non-linear projection
# -------------------------------------------------------------------------
class MultiPlane_MultiSlice_Extract_Project(nn.Module):
    """
    From 3D features X, extract multi-plane, multi-slice 2D features and
    pass through a global average pooling (ResNeXt-50 GAP) and MLP projection.

    S ∈ R^{3N×C3d×L×L} → pooled_feat ∈ R^{3N×C3d} → projection ∈ R^{3N×d}
    """

    def __init__(self, out_channels: int):
        super().__init__()
        # Use only the global average pooling layer of ResNeXt-50
        self.gap_layer = models.resnext50_32x4d(pretrained=True).avgpool

        self.non_linear_proj = nn.Sequential(
            nn.Linear(out_channels, 1536),
            nn.ReLU(inplace=True),
            nn.Linear(1536, 768),
        )

    def forward(self, input_tensor):
        # input_tensor: [B, C, L, W, H]  (L=W=H=N)

        # Coronal slices along dim=2
        coronal_slices = torch.split(input_tensor, 1, dim=2)
        Ecor = torch.cat(coronal_slices, dim=2)  # [B, C, N, W, H]

        # Sagittal slices along dim=3
        saggital_slices = torch.split(input_tensor.clone(), 1, dim=3)
        Esag = torch.cat(saggital_slices, dim=3)  # [B, C, L, N, H]

        # Axial slices along dim=4
        axial_slices = torch.split(input_tensor.clone(), 1, dim=4)
        Eax = torch.cat(axial_slices, dim=4)  # [B, C, L, W, N]

        # Elementwise product with input and plane-wise reshape
        Scor = (Ecor * input_tensor).permute(0, 2, 1, 3, 4).contiguous()  # [B, N, C, W, H]
        Ssag = (Esag * input_tensor).permute(0, 3, 1, 2, 4).contiguous()  # [B, N, C, L, H]
        Sax = (Eax * input_tensor).permute(0, 4, 1, 2, 3).contiguous()    # [B, N, C, L, W]

        # Concatenate along N dimension: [B, 3N, C, L, L]
        S = torch.cat((Scor, Ssag, Sax), dim=1)

        B, threeN, C, L, _ = S.shape
        # Merge B and 3N dimensions to feed 2D GAP
        S_2d = S.view(B * threeN, C, L, L)

        # GAP: [B*3N, C, 1, 1] -> [B*3N, C]
        pooled_feat = self.gap_layer(S_2d).squeeze(-1).squeeze(-1)

        # Non-linear projection: [B*3N, C3d] -> [B*3N, 768]
        proj = self.non_linear_proj(pooled_feat)

        # Restore [B, 3N, 768]
        output_tensor = proj.view(B, threeN, -1)
        return output_tensor


# -------------------------------------------------------------------------
# Embedding layer (cls/sep tokens + plane embeddings + positional embeddings)
# -------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    """
    Add CLS/SEP tokens, plane embeddings and positional embeddings.

    emb_size = d = 256 (or 768 in our implementation),
    total_tokens = 3S = 3 * 128 = 384  (default)
    """

    def __init__(self, emb_size: int = 256, total_tokens: int = 384):
        super().__init__()

        # z_cls ∈ R(d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # z_sep ∈ R(d)
        self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Plane-specific embeddings
        self.coronal_plane = nn.Parameter(torch.randn(1, emb_size))
        self.sagittal_plane = nn.Parameter(torch.randn(1, emb_size))
        self.axial_plane = nn.Parameter(torch.randn(1, emb_size))

        # Positional embeddings
        self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

    def forward(self, input_tensor):
        # input_tensor: [B, 3N, emb_size]
        b, n, e = input_tensor.shape
        assert n == 384, "Expected 3 * 128 tokens; got {}".format(n)

        cls_tokens = repeat(self.cls_token, "() n e -> b n e", b=b)
        sep_token = repeat(self.sep_token, "() n e -> b n e", b=b)

        # Split three planes: [0:128], [128:256], [256:384]
        x = torch.cat(
            (
                cls_tokens,
                input_tensor[:, :128, :],
                sep_token,
                input_tensor[:, 128:256, :],
                sep_token,
                input_tensor[:, 256:, :],
                sep_token,
            ),
            dim=1,
        )  # shape [B, (3N+4), emb_size]

        # Add plane embeddings in segments
        x[:, :130] += self.coronal_plane
        x[:, 130:259] += self.sagittal_plane
        x[:, 259:] += self.axial_plane

        # Add positional embeddings
        x += self.positions

        return x


# -------------------------------------------------------------------------
# Transformer blocks
# -------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(
            self.qkv(x), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3
        )
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 3, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """
    Single transformer encoder block.

    Uses:
    - LayerNorm
    - Multi-head self-attention
    - Feed-forward block
    """

    def __init__(
        self,
        emb_size: int = 256,
        drop_p: float = 0.0,
        forward_expansion: int = 3,
        forward_drop_p: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    """
    Stacked transformer encoder blocks.

    depth: number of layers.
    """

    def __init__(self, depth: int = 8, **kwargs):
        super().__init__(*[
            TransformerEncoderBlock(**kwargs)
            for _ in range(depth)
        ])


# -------------------------------------------------------------------------
# Classification head (not used directly in survival, but in backbone)
# -------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    """
    Simple linear classifier on top of CLS token.
    """

    def __init__(self, emb_size: int = 256, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        cls_token = x[:, 0]
        return self.linear(cls_token)


# -------------------------------------------------------------------------
# M3T backbone (original)
# -------------------------------------------------------------------------
class M3T(nn.Sequential):
    """
    Full M3T backbone.

    in_channels: input MRI channels
    out_channels: 3D CNN channels
    emb_size: transformer embedding size (e.g. 768)
    depth: number of transformer layers
    n_classes: final classes for the original classification head
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_size: int = 256,
        depth: int = 8,
        n_classes: int = 2,
        **kwargs,
    ):
        super().__init__(
            CNN3DBlock(in_channels, out_channels),
            MultiPlane_MultiSlice_Extract_Project(out_channels),
            EmbeddingLayer(emb_size=emb_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes),
        )


# -------------------------------------------------------------------------
# Survival wrapper: M3t
# -------------------------------------------------------------------------
class Identity(nn.Module):
    def forward(self, x):
        return x


class M3t(nn.Module):
    """
    Survival wrapper for M3T backbone.

    - Replace final classification linear layer with Identity
    - Add 4 FC heads:
        * OS risk:   fc1
        * PFS risk:  fc2
        * Age:       fc3 (with sigmoid)
        * Clinical labels (9D multi-label): fc4
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        # M3T backbone: outputs [B, emb_size]
        self.backbone = M3T(
            in_channels=3,
            out_channels=32,
            drop_p=drop_rate,
            emb_size=768,
            forward_expansion=4,
            depth=12,
        )
        # Replace classification head linear with identity
        self.backbone[-1].linear = Identity()

        self.fc1 = nn.Linear(768, 1)
        self.fc2 = nn.Linear(768, 1)
        self.fc3 = nn.Linear(768, 1)
        self.fc4 = nn.Linear(768, 9)

    def forward(self, x):
        encoded = self.backbone(x)  # [B, 768]
        os = self.fc1(encoded)
        pfs = self.fc2(encoded)
        age = torch.sigmoid(self.fc3(encoded))
        label = self.fc4(encoded)
        return os, pfs, age, label
