# Two-stage Online/Offline Phase Reconstruction (TOPR)

This repository provides an unofficial implementation of "Online Phase Reconstruction via DNN-based Phase Differences Estimation" [1].

## Licence
MIT licence.

Copyright (C) 2024 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 22.04. The verion of Python was `3.10.12`. The following modules are required:

- hydra-core
- joblib
- librosa
- numpy
- progressbar2
- pydub
- pypesq
- pystoi
- scikit-learn
- soundfile
- torch

Optionally, if you want to compare the offline version with other offline methods such as RTISI, PGHI, and SPSI, you are required to install Octave in advance. In this case, you also need to install LTFAT and PHASERET packages to compute these offline methods. Finally, the "oct2py" Python package must be installed.

## Datasets
You need to prepare the following two datasets from [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

   - basic5000: for training

   - onomatopee300: for evaluation

## Scripts

Mainly, we will use the following scripts:

| Name               | Functionality                                        |
|--------------------|------------------------------------------------------|
| config.yaml        | Configuration file in YAML format                    |
| preprocess.py      | Performs pre-processing including feature extraction |
| dataset.py         | Builds the dataset and dataloader                    |
| model.py           | Defines the network architecture                     |
| factory.py         | Defines the optimizer, scheduler, and loss function  |
| training.py        | Trains the model                                     |
| evaluate_scores.py | Calculates objective sound quality metrics           |

## Recipes

1. Prepare the two datasets form the JSUT corpus. Put those in /root_dir/trainset_dir and /root_dir/evalset_dir/, respectively.

2. Modify `config.yaml` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

3. Run `preprocess.py`. It performs preprocessing steps.

4. Run `training.py`. It performs model training.

5. Run `evaluate_scores.py`. It generates reconstructed audio data and computes objective scores (PESQ, ESTOI, LSC).

(option) 6. Run `evaluate_scores_spsi.py`, `evaluate_scores_rtisi.py`, and `evaluate_scores_rtpghi.py`. These scripts implements offline phase reconstruction methods (SPSI, RTISI, and RTPGHI). They also evaluate the objective scores (PESQ, ESTOI, LSC) to the reconstruced audios.

(option) 7. Run `plot_boxplot.py`. It plots boxplot of objective scores.

(option) 8. Run `evaluate_scores_rpu.py` and `plot_boxplot.py` if you want to compare the performance of online/offline version with RPU.


## References

[1] Y. Masuyama, K. Yatabe, K. Nagatomo and Y. Oikawa, "Online Phase Reconstruction via DNN-Based Phase Differences Estimation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 163-176, 2023.
