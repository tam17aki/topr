# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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

import math

import torch
from omegaconf import DictConfig
from torch import nn, optim


class CosineAnnealingWithWarmupLR(optim.lr_scheduler._LRScheduler):
    """Cosine Annealing with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                Instance of optimization method
            warmup_epochs (int):
                Number of epochs to perform linear warmup
            max_epochs (int):
                Number of training epochs to terminate the cosine curve
            warmup_start_lr (float):
                Learning rate at initial epoch for linear warmup
            eta_min (float):
                Lower bound of cosine curve
            last_epoch (int):
                Phase offset of cosine curve

        Schedule the learning rate along a cosine curve up to max_epochs.
        The learning curve from epoch 0 to warmup_epochs takes a linear warmup.

        Reference (in Japanese):
            https://zenn.dev/inaturam/articles/e0fa6eed17afbe
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)  # important to put here.

    def get_lr(self):
        """Return learning rate."""
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer.

    Args:
        cfg (DictConfig): configuration in YAML format.
        model (nn.Module): network parameters.
    """
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **cfg.training.optim.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: DictConfig, optimizer):
    """Instantiate scheduler.

    Args:
        cfg (DictConfig): configuration in YAML format.
        optimizer (Optimizer): Wrapped optimizer.
    """
    if cfg.training.optim.lr_scheduler.name == "CosineAnnealingWithWarmupLR":
        lr_scheduler = CosineAnnealingWithWarmupLR(
            optimizer, **cfg.training.optim.lr_scheduler.cosine_params
        )
    else:
        lr_scheduler_class = getattr(
            optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
        )
        lr_scheduler = lr_scheduler_class(
            optimizer, **cfg.training.optim.lr_scheduler.params
        )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, cfg, model):
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        self.model = model

    def _tpd2bpd(self, tpd):
        """Modify TPD to BPD.

        Args:
            tpd (Tensor): oracle backward TPD. [B, T-1, K]

        Returns:
            bpd (Tensor): oracle backward BPD. [B, T-1, K]
        """
        win_len = self.cfg.feature.win_length
        hop_len = self.cfg.feature.hop_length
        n_batch, n_frame, _ = tpd.shape
        pi_tensor = torch.Tensor([math.pi]).cuda()
        k = torch.arange(0, win_len // 2 + 1).cuda()
        angle_freq = (2 * pi_tensor / win_len) * k * hop_len
        angle_freq = angle_freq.unsqueeze(0).expand(n_frame, len(k))
        angle_freq = angle_freq.unsqueeze(0).expand(n_batch, n_frame, len(k))
        bpd = torch.angle(torch.exp(1j * (tpd - angle_freq)))
        return bpd

    def _compute_bpd_loss(self, predicted, reference):
        """Compute loss of backward baseband phase difference (BPD).

        Args:
            predicted: estimated backward BPD. [B, T-1, K]
            reference: ground-truth phase spectrum. [B, T-1, K]

        Returns:
            cosine loss of backward BPD.
        """
        oracle_tpd = reference[:, 1:, :] - reference[:, :-1, :]
        oracle_bpd = self._tpd2bpd(oracle_tpd)
        diff = predicted[:, :-1, :] - oracle_bpd
        loss = torch.sum(-torch.cos(diff), dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        return loss.mean()  # average along batch axis

    def _compute_fpd_loss(self, predicted, reference):
        """Compute loss of backward phase difference for frequency (FPD).

        Args:
            predicted: estimated backward FPD. [B, T, K]
            reference: ground-truth phase spectrum. [B, T, K]

        Returns:
            cosine loss of backward FPD.
        """
        oracle_fpd = reference[:, :, 1:] - reference[:, :, :-1]
        diff = predicted[:, :, :-1] - oracle_fpd
        loss = torch.sum(-torch.cos(diff), dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        return loss.mean()  # average along batch axis

    def forward(self, batch, mode):
        """Compute loss.

        Args:
            batch (Tuple): tuple of minibatch.

        Returns:
            loss: cosine loss of BPD or FPD.
        """
        logmag_batch, phase_batch = batch
        logmag_batch = logmag_batch.cuda().float()  # [B*T, L+1, K]
        phase_batch = phase_batch.cuda().float()  # [B, T, K]
        predicted = self.model(logmag_batch)  # [B*T, 1, K]
        predicted = predicted.squeeze()  # [B*T, K]
        predicted = predicted.reshape(
            self.cfg.training.n_batch, -1, self.cfg.feature.n_fft // 2 + 1
        )  # [B, T, K]
        if mode == "bpd":
            loss = self._compute_bpd_loss(predicted, phase_batch)
        elif mode == "fpd":
            loss = self._compute_fpd_loss(predicted, phase_batch)
        return loss


def get_loss(cfg, model):
    """Instantiate customized loss."""
    custom_loss = CustomLoss(cfg, model)
    return custom_loss
