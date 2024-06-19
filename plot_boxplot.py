# -*- coding: utf-8 -*-
"""Script for making boxplot.

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

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig


def main(cfg: DictConfig):
    """Plot boxplot of scores."""
    score_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.score_dir)
    if cfg.model.n_lookahead == 0:
        score_file = {"SPSI": None, "RTISI": None, "RTPGHI": None, "TOPR": None}
        score = {"SPSI": None, "RTISI": None, "RTPGHI": None, "TOPR": None}
    else:
        score_file = {"RTISI": None, "PGHI": None, "TOPR": None}
        score = {"RTISI": None, "PGHI": None, "TOPR": None}
    fig_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 4))

    for i, mode in enumerate(("stoi", "pesq", "lsc")):
        if cfg.model.n_lookahead == 0:
            score_file["SPSI"] = os.path.join(score_dir, f"{mode}_score_SPSI.txt")
            score_file["RTISI"] = os.path.join(score_dir, f"{mode}_score_RTISI.txt")
            score_file["RTPGHI"] = os.path.join(score_dir, f"{mode}_score_RTPGHI.txt")
            score_file["TOPR"] = os.path.join(
                score_dir, f"{mode}_score_TOPR_online.txt"
            )
            with open(score_file["SPSI"], mode="r", encoding="utf-8") as file_hander:
                score["SPSI"] = np.array([float(line.strip()) for line in file_hander])
            with open(score_file["RTISI"], mode="r", encoding="utf-8") as file_hander:
                score["RTISI"] = np.array([float(line.strip()) for line in file_hander])
            with open(score_file["RTPGHI"], mode="r", encoding="utf-8") as file_hander:
                score["RTPGHI"] = np.array(
                    [float(line.strip()) for line in file_hander]
                )
        else:
            score_file["RTISI"] = os.path.join(score_dir, f"{mode}_score_RTISI_LA.txt")
            score_file["PGHI"] = os.path.join(score_dir, f"{mode}_score_PGHI.txt")
            score_file["TOPR"] = os.path.join(
                score_dir, f"{mode}_score_TOPR_offline.txt"
            )
            with open(score_file["RTISI"], mode="r", encoding="utf-8") as file_hander:
                score["RTISI"] = np.array([float(line.strip()) for line in file_hander])
            with open(score_file["PGHI"], mode="r", encoding="utf-8") as file_hander:
                score["PGHI"] = np.array([float(line.strip()) for line in file_hander])

        with open(score_file["TOPR"], mode="r", encoding="utf-8") as file_hander:
            score["TOPR"] = np.array([float(line.strip()) for line in file_hander])

        axes = fig.add_subplot(1, 3, i + 1)

        if cfg.model.n_lookahead == 0:
            axes.boxplot(
                np.concatenate(
                    (
                        score["SPSI"].reshape(-1, 1),
                        score["RTISI"].reshape(-1, 1),
                        score["RTPGHI"].reshape(-1, 1),
                        score["TOPR"].reshape(-1, 1),
                    ),
                    axis=1,
                ),
                flierprops=dict(marker="+", markeredgecolor="r"),
                labels=["SPSI", "RTISI", "RTPGHI", "TOPR"],
                widths=(0.5, 0.5, 0.5, 0.5),
            )
        else:
            axes.boxplot(
                np.concatenate(
                    (
                        score["RTISI"].reshape(-1, 1),
                        score["PGHI"].reshape(-1, 1),
                        score["TOPR"].reshape(-1, 1),
                    ),
                    axis=1,
                ),
                flierprops=dict(marker="+", markeredgecolor="r"),
                labels=["RTISI", "PGHI", "TOPR"],
                widths=(0.5, 0.5, 0.5),
            )
        axes.xaxis.set_ticks_position("both")
        axes.yaxis.set_ticks_position("both")
        if mode == "pesq":
            axes.set_yticks([2.5, 3, 3.5, 4.0, 4.5])
        axes.tick_params(direction="in", labelsize=14)
        if mode == "lsc":
            axes.set_title(mode.upper() + " [dB]", fontsize=16)
        elif mode == "stoi":
            axes.set_title("ESTOI", fontsize=16)
        else:
            axes.set_title(mode.upper(), fontsize=16)
    fig.tight_layout()
    if cfg.model.n_lookahead == 0:
        plt.savefig(os.path.join(fig_dir, "score_online.png"))
    else:
        plt.savefig(os.path.join(fig_dir, "score_offline.png"))
    plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
