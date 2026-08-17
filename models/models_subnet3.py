# models_subnet3.py
"""
Model definition for Sub-network 3.

This file contains:
- AutoEncoder_New: a MONAI-style encoder-decoder with intermediate
  residual/conv blocks and a survival branch:
    * OS risk (scalar)
    * PFS risk (scalar)
    * Age regression (scalar, sigmoid + MSE on log(age))
    * 9D clinical multi-label logits
    * Reconstruction output (decoder)
    * Bottleneck feature (global pooled features)
"""

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["AutoEncoder_New"]


class AutoEncoder_New(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[Sequence[int]] = None,
        inter_dilations: Optional[Sequence[int]] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
    ) -> None:
        """
        MONAI-based autoencoder with an additional survival branch.

        Args:
            dimensions: 2D or 3D convolutions (here we use 3D).
            in_channels: number of input channels.
            out_channels: number of output channels for decoder.
            channels: encoder feature channels.
            strides: encoder strides.
            kernel_size: encoder kernel size.
            up_kernel_size: decoder kernel size.
            num_res_units: number of residual units per encode/decode block.
            inter_channels: list of channels in the intermediate residual / conv blocks.
            inter_dilations: list of dilations for intermediate blocks.
            num_inter_units: number of subunits per intermediate residual unit.
            act, norm, dropout, bias: standard MONAI block options.
        """
        super().__init__()

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.num_inter_units = num_inter_units
        self.inter_channels = list(inter_channels or [])
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        # Encoder / intermediate / decoder
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.intermediate, self.encoded_channels = self._get_intermediate_module(
            self.encoded_channels, self.num_inter_units
        )
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])

        # Survival branch
        self.survivalnet = self._get_survival_module(self.encoded_channels)
        self.fc1 = nn.Linear(1024, 1)  # OS risk
        self.fc2 = nn.Linear(1024, 1)  # PFS risk
        self.fc3 = nn.Linear(1024, 1)  # Age regression
        self.fc4 = nn.Linear(1024, 9)  # 9 clinical multi-label logits

        # Bottleneck global pooling to get compact feature representation
        self.bn = nn.Sequential(
            nn.AvgPool3d(8),
            nn.Flatten(),
        )

    def _get_survival_module(self, in_channels: int):
        survNet = nn.Sequential()

        tmp = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=512, out_channels=1024, kernel_size=2, stride=1),
            nn.BatchNorm3d(1024),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),
            nn.BatchNorm3d(1024),
            nn.PReLU(),
            nn.Dropout3d(self.dropout),
            nn.Flatten(),
        )

        survNet.add_module("surv_FCN", tmp)

        return survNet


    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module(f"encode_{i}", layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: int) -> Tuple[nn.Module, int]:
        intermediate: nn.Module = nn.Identity()
        layer_channels = in_channels

        if self.inter_channels:
            intermediate = nn.Sequential()
            for i, (dc, di) in enumerate(zip(self.inter_channels, self.inter_dilations)):
                if self.num_res_units > 0:
                    unit = ResidualUnit(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        subunits=num_inter_units,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )
                else:
                    unit = Convolution(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                    )
                intermediate.add_module(f"inter_{i}", unit)
                layer_channels = dc

        return intermediate, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module(f"decode_{i}", layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:
        if self.num_res_units > 0:
            return ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
        return Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:
        decode = nn.Sequential()

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )
        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
            decode.add_module("resunit", ru)

        return decode

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass.

        Returns:
            os (Tensor): OS risk.
            pfs (Tensor): PFS risk.
            age (Tensor): age regression output (after sigmoid).
            label (Tensor): 9D clinical multi-label logits.
            x_rec (Tensor): reconstructed image from decoder.
            f (Tensor): bottleneck feature vector after global pooling.
        """
        x = self.encode(x)
        encoded = self.intermediate(x)
        x_rec = self.decode(encoded)

        x1 = self.survivalnet(encoded)
        os = self.fc1(x1)
        pfs = self.fc2(x1)
        age = torch.sigmoid(self.fc3(x1))
        label = self.fc4(x1)

        # bottleneck feature (for downstream analysis)
        f = self.bn(encoded)
        return os, pfs, age, label, x_rec, f
