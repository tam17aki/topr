# -*- coding: utf-8 -*-
"""Evaluation script for sound quality based on PESQ, STOI and LSC.

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

import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from pesq import pesq
from progressbar import progressbar as prg
from pystoi import stoi
from scipy import signal
from scipy.sparse import csr_array, diags_array
from scipy.sparse.linalg import spsolve
from torch.multiprocessing import set_start_method

from model import get_model


def load_checkpoint(cfg: DictConfig):
    """Load checkpoint.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        model_bpd (nn.Module): DNNs to estimate BPD.
        model_fpd (nn.Module): DNNs to estimate FPD.
    """
    model_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.model_dir)
    model_bpd = get_model(cfg)
    model_file = os.path.join(model_dir, cfg.training.model_file + ".bpd.pth")
    checkpoint = torch.load(model_file)
    model_bpd.load_state_dict(checkpoint)

    model_fpd = get_model(cfg)
    model_file = os.path.join(model_dir, cfg.training.model_file + ".fpd.pth")
    checkpoint = torch.load(model_file)
    model_fpd.load_state_dict(checkpoint)
    return model_bpd, model_fpd


def get_wavdir(cfg):
    """Return dirname of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    if cfg.model.n_lookahead == 0:
        wav_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "online", "TOPR")
    else:
        wav_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "offline", "TOPR")
    return wav_dir


def get_wavname(cfg, basename):
    """Return filename of wavefile to be evaluated.

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("-")[0]
    wav_dir = get_wavdir(cfg)
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(cfg, basename):
    """Compute PESQ and wideband PESQ.

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.TOPR.root_dir,
        cfg.TOPR.data_dir,
        cfg.TOPR.evalset_dir,
        cfg.TOPR.resample_dir,
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return pesq(rate, reference, eval_wav, cfg.demo.pesq_band)


def compute_stoi(cfg, basename):
    """Compute STOI or extended STOI (ESTOI).

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.TOPR.root_dir,
        cfg.TOPR.data_dir,
        cfg.TOPR.evalset_dir,
        cfg.TOPR.resample_dir,
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    return stoi(reference, eval_wav, rate, extended=cfg.demo.stoi_extended)


def compute_lsc(cfg, basename):
    """Compute log-spectral convergence (LSC).

    Args:
        cfg (DictConfig): configuration.
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: log-spectral convergence.
    """
    eval_wav, _ = sf.read(get_wavname(cfg, basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]
    wav_dir = os.path.join(
        cfg.TOPR.root_dir,
        cfg.TOPR.data_dir,
        cfg.TOPR.evalset_dir,
        cfg.TOPR.resample_dir,
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if len(eval_wav) > len(reference):
        eval_wav = eval_wav[: len(reference)]
    else:
        reference = reference[: len(eval_wav)]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=rate,
        mfft=cfg.feature.n_fft,
    )
    ref_abs = np.abs(stfft.stft(reference))
    eval_abs = np.abs(stfft.stft(eval_wav))
    lsc = np.linalg.norm(ref_abs - eval_abs)
    lsc = lsc / np.linalg.norm(ref_abs)
    lsc = 20 * np.log10(lsc)
    return lsc


def bpd2tpd(bpd, win_len, hop_len):
    """Convert BPD to TPD.

    Args:
        bpd (ndarray): BPD.
        win_len (int): length of analysis window.
        hop_len (int): length of window shift.

    Returns:
        tpd (ndarray): TPD.
    """
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * hop_len
    tpd = bpd + angle_freq
    return tpd


@torch.no_grad()
def compute_1st_stage(cfg, model_tuple, logmag):
    """Estimate TPD and FPD from log-magnitude spectra.

    Args:
        model_tuple (tuple): tuple of DNN params (nn.Module).
        logmag (ndarray): log magnitude spectrum. [1, L, K]

    Returns:
        tpd (ndarray): TPD. [K]
        fpd (ndarray): FPD. [K-1]
    """
    model_bpd, model_fpd = model_tuple  # DNNs
    bpd = model_bpd(logmag)  # [1, 1, K]
    fpd = model_fpd(logmag)  # [1, 1, K]
    bpd = bpd.cpu().detach().numpy().copy().squeeze()
    fpd = fpd.cpu().detach().numpy().copy().squeeze()
    fpd = fpd[:-1]
    tpd = bpd2tpd(bpd, cfg.feature.win_length, cfg.feature.hop_length)
    return tpd, fpd


def compute_2nd_stage(cfg, phase_prev, pd_tuple, mag_cur, mag_prev):
    """Reconstruct phase spectrum.

    Args:
        cfg (DictConfig): configuration.
        phase_prev (ndarray): phase spectrum at the previous frame. [K]
        pd_tuple (Tuple): tuple of TPD and FPD (ndarray).
        mag_cur (ndarray): magnitude spectrum at the current frame. [K]
        mag_prev (ndarray): magnitude spectrum at the previous frame. [K]

    Returns:
        phase (ndarray): reconstructed phase spectrum at the current frame. [K]
    """
    var = {"coef": None, "rhs": None, "sol": None}
    tpd, fpd = pd_tuple
    n_fbin = mag_cur.shape[0]

    # complex ratios
    ratio_u = mag_cur[1:] / mag_cur[:-1]  # [K-1]
    ratio_u = ratio_u * np.exp(1j * fpd)  # [K-1]  Eqs. (37) and (41)

    # weight matrix (diagonal)
    lambda_vec = (mag_cur * mag_prev) ** cfg.demo.weight_power
    gamma_mat = cfg.demo.weight_gamma * (
        (mag_cur[1:] * mag_cur[:-1]) ** cfg.demo.weight_power
    )
    gamma_mat = diags_array(gamma_mat, format="csr")
    lambda_mat = diags_array(lambda_vec, format="csr")

    d_mat = csr_array(
        (
            np.append(-1.0 * ratio_u, np.ones(n_fbin - 1)),  # data
            (
                list(range(n_fbin - 1)) + list(range(n_fbin - 1)),
                list(range(n_fbin - 1)) + list(range(1, n_fbin)),
            ),  # (row_ind, col_ind)
        ),
        shape=(n_fbin - 1, n_fbin),
        dtype=np.complex128,
    )  # Eqs. (44) and (45)
    var["coef"] = lambda_mat + d_mat.T.tocsr() @ gamma_mat @ d_mat
    var["rhs"] = lambda_vec * mag_cur * np.exp(1j * (phase_prev + tpd))
    var["sol"] = spsolve(var["coef"], var["rhs"])
    phase = np.angle(var["sol"])  # Eq. (46)
    return phase


def reconst_phase(cfg, model_tuple, logmag, magnitude):
    """Reconstruct phase spectrum by TOPR algorithm.

    Y. Masuyama, K. Yatabe, K. Nagatomo and Y. Oikawa,
    "Online Phase Reconstruction via DNN-Based Phase Differences Estimation,"
    in IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    vol. 31, pp. 163-176, 2023, doi: 10.1109/TASLP.2022.3221041.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNNs (nn.Module).
        logmag (Tensor): log-magnitude spectrum (zero padded). [T, K]
        magnitude (Tensor): magnitude spectrum. [T, K]

    Returns:
        phase (ndarray): reconstruced phase. [T, K]
    """
    logmag = np.pad(
        logmag, ((cfg.model.n_lookback, cfg.model.n_lookahead), (0, 0)), "constant"
    )
    logmag = torch.from_numpy(logmag).float().unsqueeze(0).cuda()  # [1, T+L+1, K]
    n_frame, n_fbin = magnitude.shape
    n_lookback = cfg.model.n_lookback
    n_lookahead = cfg.model.n_lookahead

    phase = np.zeros((n_frame, n_fbin))  # [T, K]
    _, fpd = compute_1st_stage(
        cfg, model_tuple, logmag[:, : n_lookback + n_lookahead + 1, :]
    )
    for k in range(1, n_fbin):
        phase[0, k] = phase[0, k - 1] + fpd[k - 1]
    for i in range(1, n_frame):
        tpd, fpd = compute_1st_stage(
            cfg, model_tuple, logmag[:, i : i + n_lookback + n_lookahead + 1, :]
        )  # Eqs. (29), (30)
        phase[i, :] = compute_2nd_stage(
            cfg, phase[i - 1, :], (tpd, fpd), magnitude[i, :], magnitude[i - 1, :]
        )  # Eq. (31)
    return phase


def _reconst_waveform(cfg, model_tuple, logmag_path):
    """Reconstruct audio waveform only from the magnitude spectrum.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_path (str): path to the log-magnitude spectrum.

    Returns:
        None.
    """
    logmag = np.load(logmag_path)  # [T, K]
    magnitude = np.exp(logmag)  # [T, K]
    phase = reconst_phase(cfg, model_tuple, logmag, magnitude)  # [T, K]
    reconst_spec = magnitude * np.exp(1j * phase)  # [T, K]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=cfg.feature.sample_rate,
        mfft=cfg.feature.n_fft,
    )
    audio = stfft.istft(reconst_spec.T)
    wav_file = get_wavname(cfg, os.path.basename(logmag_path))
    sf.write(wav_file, audio, cfg.feature.sample_rate)


def reconst_waveform(cfg, model_tuple, logmag_list):
    """Reconstruct audio waveforms in parallel.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_list (list): list of path to the log-magnitude spectrum.

    Returns:
        None.
    """
    set_start_method("spawn")
    with ProcessPoolExecutor(cfg.preprocess.n_jobs) as executor:
        futures = [
            executor.submit(_reconst_waveform, cfg, model_tuple, logmag_path)
            for logmag_path in logmag_list
        ]
        for future in prg(
            futures, prefix="Reconstruct waveform: ", suffix=" ", redirect_stdout=False
        ):
            future.result()  # return None


def compute_obj_scores(cfg, logmag_list):
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        cfg (DictConfig): configuration.
        logmag_list (list): list of path to the log-magnitude spectrum.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict = {"pesq": [], "stoi": [], "lsc": []}
    for logmag_path in prg(
        logmag_list,
        prefix="Compute objective scores: ",
        suffix=" ",
        redirect_stdout=False,
    ):
        score_dict["pesq"].append(compute_pesq(cfg, os.path.basename(logmag_path)))
        score_dict["stoi"].append(compute_stoi(cfg, os.path.basename(logmag_path)))
        score_dict["lsc"].append(compute_lsc(cfg, os.path.basename(logmag_path)))
    return score_dict


def aggregate_scores(cfg, score_dict, score_dir):
    """Aggregate objective evaluation scores.

    Args:
        cfg (DictConfig): configuration.
        score_dict (dict): dictionary of objective score lists.
        score_dir (str): dictionary name of objective score files.

    Returns:
        None.
    """
    for score_type, score_list in score_dict.items():
        if cfg.model.n_lookahead == 0:
            out_filename = f"{score_type}_score_TOPR_online.txt"
        else:
            out_filename = f"{score_type}_score_TOPR_offline.txt"
        out_filename = os.path.join(score_dir, out_filename)
        with open(out_filename, mode="w", encoding="utf-8") as file_handler:
            for score in score_list:
                file_handler.write(f"{score}\n")
        score_array = np.array(score_list)
        print(
            f"{score_type}: "
            f"mean={np.mean(score_array):.6f}, "
            f"median={np.median(score_array):.6f}, "
            f"std={np.std(score_array):.6f}, "
            f"max={np.max(score_array):.6f}, "
            f"min={np.min(score_array):.6f}"
        )


def main(cfg: DictConfig):
    """Perform evaluation."""
    # setup directory
    wav_dir = get_wavdir(cfg)
    os.makedirs(wav_dir, exist_ok=True)
    score_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.score_dir)
    os.makedirs(score_dir, exist_ok=True)

    # load DNN parameters
    model_bpd, model_fpd = load_checkpoint(cfg)
    model_bpd.cuda()
    model_fpd.cuda()
    model_bpd.eval()
    model_fpd.eval()

    # load log-magnitude spectra
    feat_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.feat_dir, cfg.TOPR.evalset_dir, cfg.feature.window
    )
    logmag_list = glob.glob(feat_dir + "/*-feats_logmag.npy")
    logmag_list.sort()

    # reconstruct phase and waveform
    reconst_waveform(cfg, (model_bpd, model_fpd), logmag_list)

    # compute objective scores
    score_dict = compute_obj_scores(cfg, logmag_list)

    # aggregate objective scores
    aggregate_scores(cfg, score_dict, score_dir)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
