# -*- coding: utf-8 -*-
"""Model definition of Two-stage Online/Offline Phase Reconstruction (TOPR).

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn


class MeanSubtraction(nn.Module):
    """Mean subtraction layer."""

    def __init__(self):
        """Initialize class."""
        super().__init__()

    def forward(self, inputs):
        """Perform mean subtraction.

        Args:
            inputs: log-magnitude. [B, L+1, K]

        Returns:
            outputs: mean subtracted log-magnitude. [B, L+1, K]
        """
        outputs = inputs - torch.mean(inputs, dim=1, keepdim=True)
        return outputs


class FreqConv(nn.Module):
    """1-D Frequency convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size):
        """Initialize class.

        Args:
            in_channels (int): # of input channels.
            out_channels (int): # of output channels.
            kernel_size (int): kernel size of 1-D freq conv.
        """
        super().__init__()
        self.freq_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs (Tensor): T-F features. [B, C_in, K]

        Returns:
            outputs (Tensor): T-F features. [B, C_out, K]
        """
        outputs = self.freq_conv(inputs)
        return outputs


class FreqGatedConv(nn.Module):
    """1-D frequency gated convolution layer."""

    def __init__(self, n_channels, kernel_size):
        """Initialize class.

        Args:
            n_channels (int): # of channels.
            kernel_size (int): kernel size of freq gated conv.
        """
        super().__init__()
        self.freq_conv1 = FreqConv(n_channels, n_channels, kernel_size)
        self.freq_conv2 = FreqConv(n_channels, n_channels, kernel_size)
        self.gate = nn.Sigmoid()

    def forward(self, inputs):
        """Forward propagation."""
        return self.freq_conv1(inputs) * self.gate(self.freq_conv2(inputs))


class ResidualBlock(nn.Module):
    """Residual block module.

    This consists of two FreqGatedConv modules.
    """

    def __init__(self, n_channels, kernel_size):
        """Initialize class.

        Args:
            n_channels (int): # of channels.
            kernel_size (int): kernel size of freq gated conv.
        """
        super().__init__()
        self.freq_gated1 = FreqGatedConv(n_channels, kernel_size)
        self.freq_gated2 = FreqGatedConv(n_channels, kernel_size)

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs: T-F features. [B, C, K]

        Returns:
            outputs: T-F features. [B, C, K]
        """
        hidden = self.freq_gated1(inputs)
        outputs = inputs + self.freq_gated2(hidden)
        return outputs


class TOPRNet(nn.Module):
    """DNN for Two-stage Online/Offline Phase Reconstruction."""

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        n_lookahead = config.model.n_lookahead
        n_lookback = config.model.n_lookback
        n_channels = config.model.n_channels
        kernel_size = config.model.kernel_size
        self.net = nn.Sequential(
            MeanSubtraction(),
            FreqConv(n_lookback + n_lookahead + 1, n_channels, kernel_size=1),
            FreqGatedConv(n_channels, kernel_size),
            ResidualBlock(n_channels, kernel_size),
            ResidualBlock(n_channels, kernel_size),
            FreqConv(n_channels, 1, kernel_size=1),
        )

    def forward(self, inputs):
        """Forward propagation.

        Args:
            inputs: log-magnitude spectra. [B, L+1, K]

        Returns:
            outputs: BPD or FPD. [B, 1, K]
        """
        outputs = self.net(inputs)
        return outputs


def get_model(cfg):
    """Instantiate models.

    Args:
        cfg (DictConfig): configuration in YAML format.
        device: device info.
    """
    model = TOPRNet(cfg)
    return model
