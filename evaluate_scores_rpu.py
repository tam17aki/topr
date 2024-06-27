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
from scipy.linalg import solve_banded
from scipy.sparse import csr_array
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
        if cfg.demo.weighted_rpu is True:
            wav_dir = os.path.join(
                cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "online", "wRPU"
            )
        else:
            wav_dir = os.path.join(
                cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "online", "RPU"
            )
    else:
        if cfg.demo.weighted_rpu is True:
            wav_dir = os.path.join(
                cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "offline", "wRPU"
            )
        else:
            wav_dir = os.path.join(
                cfg.TOPR.root_dir, cfg.TOPR.demo_dir, "offline", "RPU"
            )
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
    return pesq(rate, reference, eval_wav)


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


def bpd2tpd(bpd, win_len, hop_len, n_frames):
    """Convert BPD to TPD.

    Args:
        bpd (ndarray): backward BPD. [K]
        win_len (int): length of analysis window.
        hop_len (int): length of window shift.
        n_frames (int): number of frames.

    Returns:
        tpd (ndarray): backward TPD. [K]
    """
    k = np.arange(0, win_len // 2 + 1)
    angle_freq = (2 * np.pi / win_len) * k * hop_len
    angle_freq = np.tile(np.expand_dims(angle_freq, 0), [n_frames, 1])
    tpd = bpd + angle_freq
    return tpd


@torch.no_grad()
def compute_1st_stage(cfg, model_tuple, logmag):
    """Estimate backward TPD and FPD from log-magnitude spectra.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNNs (nn.Module).
        logmag (Tensor): log-magnitude spectra. [T, L, K]

    Returns:
        tpd (ndarray): backward TPD. [T, K]
        fpd (ndarray): backward FPD. [T, K-1]
    """
    logmag = logmag.unfold(
        1, cfg.model.n_lookback + cfg.model.n_lookahead + 1, 1
    )  # [1, T, K, L]
    _, n_frame, n_fbin, width = logmag.shape  # [1, T, K, L]
    logmag = logmag.reshape(-1, n_fbin, width)  # [T, K, L]
    logmag = logmag.transpose(2, 1)  # [T, L, K]
    model_bpd, model_fpd = model_tuple  # DNNs
    bpd = model_bpd(logmag)  # [T, 1, K]
    fpd = model_fpd(logmag)  # [T, 1, K]
    bpd = bpd.cpu().detach().numpy().copy().squeeze()  # [T, K]
    fpd = fpd.cpu().detach().numpy().copy().squeeze()  # [T, K]
    tpd = bpd2tpd(bpd, cfg.feature.win_length, cfg.feature.hop_length, n_frame)
    return tpd, fpd[:, :-1]  # [T, K], [T, K-1]


def get_band_coef(matrix):
    """Return band tridiagonal elements of coef matrix.

    Args:
        matrix (ndarray): band tridiagonal matrix.

    Returns:
        band_elem (ndarray): band tridiagonal elements (upper, diag, and lower).
    """
    upper = np.diag(matrix, 1)
    upper = np.concatenate((np.array([0]), upper))
    lower = np.diag(matrix, -1)
    lower = np.concatenate((lower, np.array([0])))
    band_elem = np.concatenate(
        (upper.reshape(1, -1), np.diag(matrix).reshape(1, -1), lower.reshape(1, -1))
    )
    return band_elem


def wrap_phase(phase):
    """Compute wrapped phase.

    Args:
        phase (ndarray): phase spectrum.

    Returns:
        wrapped phase (ndarray).
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def compute_rpu(ifreq, grd, magnitude, weighted_rpu=False, weight_power=5):
    """Reconstruct phase by Recurrent Phase Unwrapping (RPU).

    This function performs phase reconstruction via RPU.

    Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada,
    Phase reconstruction based on recurrent phase unwrapping with deep neural
    networks, IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), May 2020.

    For weighted RPU, see:

    N. B. Thien, Y. Wakabayashi, K. Iwai and T. Nishiura,
    Inter-Frequency Phase Difference for Phase Reconstruction Using Deep Neural
    Networks and Maximum Likelihood, in IEEE/ACM Transactions on Audio,
    Speech, and Language Processing, vol. 31, pp. 1667-1680, 2023.

    Args:
        ifreq (ndarray): instantaneous frequency. [T-1, K]
        grd   (ndarray): group delay. [T, K-1]
        magnitude (ndarray): magnitude spectrum. [T, K]
        weighted_rpu (bool): flag to apply weighted RPU.
        weight_power (int): power to weight.

    Returns:
        phase (ndarray): reconstructed phase. [T, K]
    """
    n_frame, n_feats = magnitude.shape
    phase = np.zeros_like(magnitude)
    fd_mat = (  # frequency-directional differential operator (matrix)
        -np.triu(np.ones((n_feats - 1, n_feats)), 1)
        + np.triu(np.ones((n_feats - 1, n_feats)), 2)
        + np.eye(n_feats - 1, n_feats)
    )
    fd_mat = csr_array(fd_mat)
    var = {"ph_temp": None, "dwp": None, "fdd_coef": None, "coef": None, "rhs": None}
    for k in range(1, n_feats):
        phase[0, k] = phase[0, k - 1] - grd[0, k - 1]
    if weighted_rpu is False:
        var["coef"] = fd_mat.T @ fd_mat + np.eye(n_feats)
        var["coef"] = get_band_coef(var["coef"])
        for i in range(1, n_frame):
            var["ph_temp"] = wrap_phase(phase[i - 1, :]) + ifreq[i - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[i, :] - var["dwp"])
            var["rhs"] = var["ph_temp"] + fd_mat.T @ grd_new
            phase[i, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    else:
        for i in range(1, n_frame):
            w_ifreq = magnitude[i - 1, :] ** weight_power
            w_grd = magnitude[i, :-1] ** weight_power
            var["fdd_coef"] = fd_mat.T * w_grd
            var["coef"] = np.diag(w_ifreq) + var["fdd_coef"] @ fd_mat
            var["coef"] = get_band_coef(var["coef"])
            var["ph_temp"] = wrap_phase(phase[i - 1, :]) + ifreq[i - 1, :]
            var["dwp"] = fd_mat @ var["ph_temp"]
            grd_new = var["dwp"] + wrap_phase(grd[i, :] - var["dwp"])
            var["rhs"] = w_ifreq * var["ph_temp"] + var["fdd_coef"] @ grd_new
            phase[i, :] = solve_banded((1, 1), var["coef"], var["rhs"])
    return phase


def _reconst_waveform(cfg, model_tuple, logmag_path):
    """Reconstruct audio waveform only from the magnitude spectra.

    Notice that the instantaneous frequency and group delay are estimated
    by the 1st stage of TOPR, respectively.

    The phase spectrum is reconstruced via RPU.

    Args:
        cfg (DictConfig): configuration.
        model_tuple (Tuple): tuple of DNN params (nn.Module).
        logmag_path (str): path to the log-magnitude spectrum.

    Returns:
        None.
    """
    logmag = np.load(logmag_path)  # [T, K]
    magnitude = np.exp(logmag)  # [T, K]

    # estimate TPD and FPD from log-magnitude spectra.
    logmag = np.pad(
        logmag, ((cfg.model.n_lookback, cfg.model.n_lookahead), (0, 0)), "constant"
    )
    logmag = torch.from_numpy(logmag).float().unsqueeze(0).cuda()  # [1, T+L+1, K]
    tpd, fpd = compute_1st_stage(cfg, model_tuple, logmag)

    # reconstruct phase spectra by using instantaneous frequency (= TPD) and
    # group delay (= negative FPD).
    phase = compute_rpu(
        tpd, -fpd, magnitude, cfg.demo.weighted_rpu, cfg.demo.weight_power_rpu
    )

    # reconstruct audio waveform
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


def compute_eval_score(cfg, logmag_list):
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
            if cfg.demo.weighted_rpu is True:
                out_filename = f"{score_type}_score_wRPU_online.txt"
            else:
                out_filename = f"{score_type}_score_RPU_online.txt"
        else:
            if cfg.demo.weighted_rpu is True:
                out_filename = f"{score_type}_score_wRPU_offline.txt"
            else:
                out_filename = f"{score_type}_score_RPU_offline.txt"
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
    score_dict = compute_eval_score(cfg, logmag_list)

    # aggregate objective scores
    aggregate_scores(cfg, score_dict, score_dir)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
