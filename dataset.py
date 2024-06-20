# -*- coding: utf-8 -*-
"""Dataset definition for Two-stage Online/Offline Phase Reconstruction (TOPR).

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

import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TOPRDataset(Dataset):
    """Dataset for TOPR."""

    def __init__(self, feat_paths):
        """Initialize class."""
        self.logmag_paths = feat_paths["logmag"]
        self.phase_paths = feat_paths["phase"]

    def __getitem__(self, idx):
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return (np.load(self.logmag_paths[idx]), np.load(self.phase_paths[idx]))

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return len(self.logmag_paths)


def collate_fn_topr(batch, cfg):
    """Collate function for TOPR.

    Args:
        batch (Tuple): tuple of minibatch.
        cfg (DictConfig): configuration in YAML format.

    Returns:
        tuple: a batch of inputs and targets.
    """
    batch_feats = {"logmag": None, "phase": None}
    for j, feat in enumerate(batch_feats.keys()):
        batch_temp = [x[j] for x in batch]
        batch_feats[feat] = torch.from_numpy(np.array(batch_temp))
        if feat == "logmag":
            batch_feats[feat] = batch_feats[feat].unfold(
                1, cfg.model.n_lookback + cfg.model.n_lookahead + 1, 1
            )
            _, _, n_fbin, width = batch_feats[feat].shape
            batch_feats[feat] = batch_feats[feat].reshape(-1, n_fbin, width)
            batch_feats[feat] = batch_feats[feat].transpose(2, 1)
        else:
            _, n_frame, _ = batch_feats[feat].shape
            batch_feats[feat] = batch_feats[feat][
                :, cfg.model.n_lookback : n_frame - cfg.model.n_lookahead, :
            ]

    return (batch_feats["logmag"], batch_feats["phase"])


def get_dataloader(cfg):
    """Get data loaders for training and validation.

    Args:
        cfg (DictConfig): configuration in YAML format.

    Returns:
        dict: Data loaders.
    """
    wav_list = os.listdir(
        os.path.join(
            cfg.TOPR.root_dir,
            cfg.TOPR.data_dir,
            cfg.TOPR.trainset_dir,
            cfg.TOPR.split_dir,
        )
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.feat_dir, cfg.TOPR.trainset_dir, cfg.feature.window
    )
    feat_paths = {"logmag": None, "phase": None}
    for feat in feat_paths:
        feat_paths[feat] = [
            os.path.join(feat_dir, f"{utt_id}-feats_{feat}.npy") for utt_id in utt_list
        ]

    data_loaders = DataLoader(
        TOPRDataset(feat_paths),
        batch_size=cfg.training.n_batch,
        collate_fn=functools.partial(collate_fn_topr, cfg=cfg),
        pin_memory=True,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=True,
    )
    return data_loaders
