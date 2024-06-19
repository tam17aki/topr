# -*- coding: utf-8 -*-
"""Training script for Two-stage Online/Offline Phase Reconstruction (TOPR).

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

import os
from collections import namedtuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from progressbar import progressbar as prg
from torch import nn
from torchinfo import summary

from dataset import get_dataloader
from factory import get_loss, get_lr_scheduler, get_optimizer
from model import get_model


def get_training_modules(cfg: DictConfig):
    """Instantiate modules for training."""
    dataloader = get_dataloader(cfg)
    model = get_model(cfg).cuda()
    loss_func = get_loss(cfg, model)
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = None
    if cfg.training.use_scheduler:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    TrainingModules = namedtuple(
        "TrainingModules",
        ["dataloader", "model", "loss_func", "optimizer", "lr_scheduler"],
    )
    modules = TrainingModules(dataloader, model, loss_func, optimizer, lr_scheduler)
    summary(model)
    return modules


def training_loop(cfg: DictConfig, modules, mode):
    """Perform training loop."""
    dataloader, model, loss_func, optimizer, lr_scheduler = modules
    model.train()
    n_epoch = cfg.training.n_epoch + 1
    for epoch in prg(
        range(1, n_epoch), prefix="Model training: ", suffix=" ", redirect_stdout=False
    ):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func.forward(batch, mode)
            epoch_loss += loss.item()
            loss.backward()
            if cfg.training.use_grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_max_norm)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        epoch_loss = epoch_loss / len(dataloader)
        if epoch == 1 or epoch % cfg.training.report_interval == 0:
            print(f"\nEpoch {epoch}: loss = {epoch_loss:.12f} ")


def save_checkpoint(cfg: DictConfig, modules, mode):
    """Save checkpoint."""
    model = modules.model
    model_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    if mode == "bpd":
        model_file = os.path.join(model_dir, cfg.training.model_file + ".bpd.pth")
    elif mode == "fpd":
        model_file = os.path.join(model_dir, cfg.training.model_file + ".fpd.pth")
    torch.save(model.state_dict(), model_file)


def main(cfg: DictConfig):
    """Perform model training."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    modules = get_training_modules(cfg)
    training_loop(cfg, modules, "bpd")
    save_checkpoint(cfg, modules, "bpd")

    modules = get_training_modules(cfg)
    training_loop(cfg, modules, "fpd")
    save_checkpoint(cfg, modules, "fpd")


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
