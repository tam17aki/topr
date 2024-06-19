# -*- coding: utf-8 -*-
"""Proprocess script: resampling, trimming, and feature extraction.

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
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf
from hydra import compose, initialize
from progressbar import progressbar as prg
from pydub import AudioSegment
from scipy import signal


def get_basic5000_label(cfg):
    """Download label data for BASIC5000 from sarulab-repo.

    Args:
        cfg (DictConfig): configuration in YAML format.
    """
    repo_url = cfg.preprocess.repo_url
    label_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.label_dir)
    os.makedirs(label_dir, exist_ok=True)
    subprocess.run(
        "echo -n Downloading labels of BASIC5000 ...", text=True, shell=True, check=True
    )

    command = "wget " + "-P " + "/tmp/" + " " + repo_url
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = "cd " + "/tmp/" + "; " + "unzip " + "master.zip"
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)

    command = (
        "cp "
        + os.path.join("/tmp/", cfg.preprocess.repo_name, "labels/basic5000/*.lab")
        + " "
        + os.path.join(label_dir)
    )
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = "rm " + "/tmp/master.zip"
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    command = "rm -rf " + os.path.join("/tmp/", cfg.preprocess.repo_name)
    subprocess.run(command, text=True, shell=True, capture_output=True, check=True)
    print(" done.")


def resample_wav(cfg, sample_rate=44100, is_train=True):
    """Resample wav file.

    Notice:
        the original audios must be put in the dataset_dir/"orig",
        e.g., /work/tamamori/topr/data/basic5000/orig/

    Args:
        cfg (DictConfig): configuration in YAML format.
        sample_rate (int): sampling rate of the original audio.
        is_train (Bool): handling training dataset or test dataset.
    """
    dataset_dir = cfg.TOPR.trainset_dir if is_train is True else cfg.TOPR.evalset_dir
    wav_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.data_dir, dataset_dir, "orig")
    resample_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.data_dir, dataset_dir, cfg.TOPR.resample_dir
    )
    wav_list = os.listdir(wav_dir)  # basename
    wav_list.sort()

    os.makedirs(resample_dir, exist_ok=True)
    for wav_name in prg(
        wav_list, prefix="Resampling wave files: ", suffix=" ", redirect_stdout=False
    ):
        wav_path = os.path.join(wav_dir, wav_name)
        wav, _ = librosa.load(wav_path, sr=sample_rate)
        down_sampled = librosa.resample(
            wav, orig_sr=sample_rate, target_sr=cfg.preprocess.resample_rate
        )
        out_path = os.path.join(resample_dir, wav_name)
        sf.write(out_path, down_sampled, cfg.preprocess.resample_rate, subtype="PCM_16")


def trim_silence(cfg, is_train=True):
    """Remove silence by using label info and save the result as wav.

    Args:
        cfg (DictConfig): configuration in YAML format.
        is_train (Bool): handling training dataset or test dataset.
    """
    label_dir = os.path.join(cfg.TOPR.root_dir, cfg.TOPR.label_dir)
    label_list = glob.glob(label_dir + "/*.lab")
    label_list.sort()

    dataset_dir = cfg.TOPR.trainset_dir if is_train is True else cfg.TOPR.evalset_dir
    resample_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.data_dir, dataset_dir, cfg.TOPR.resample_dir
    )
    trim_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.data_dir, dataset_dir, cfg.TOPR.resample_trim_dir
    )
    os.makedirs(trim_dir, exist_ok=True)

    for lab_file in prg(
        label_list, prefix="Trimming silence: ", suffix=" ", redirect_stdout=False
    ):
        with open(lab_file, mode="r", encoding="utf-8") as file_handler:
            lines = file_handler.read().splitlines()
            begin_info = lines[0].split()[1]  # start time of the first segment
            begin_info = float(begin_info) / 10**7  # conversion from 10^7 sec to ms
            end_info = lines[-2].split()[1]  # start time of the last segment
            end_info = float(end_info) / 10**7  # conversion from 10^7 sec to ms
            file_name, _ = os.path.splitext(os.path.basename(lab_file))
            wav_file = os.path.join(resample_dir, file_name + ".wav")
            audio = AudioSegment.from_wav(wav_file)
            audio = audio[begin_info * 1000 : end_info * 1000]  # ms -> sec
            audio.export(
                os.path.join(trim_dir, os.path.basename(wav_file)), format="wav"
            )


def split_utterance(cfg):
    """Split utterances after resampling into segments.

    Args:
        cfg (DictConfig): configuration in YAML format.
    """
    wav_dir = os.path.join(
        cfg.TOPR.root_dir,
        cfg.TOPR.data_dir,
        cfg.TOPR.trainset_dir,
        cfg.TOPR.resample_trim_dir,
    )
    wav_list = glob.glob(wav_dir + "/*.wav")
    wav_list.sort()
    sec_per_split = cfg.preprocess.sec_per_split

    out_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.data_dir, cfg.TOPR.trainset_dir, cfg.TOPR.split_dir
    )
    os.makedirs(out_dir, exist_ok=True)
    for wav_name in prg(
        wav_list, prefix="Split utterances: ", suffix=" ", redirect_stdout=False
    ):
        audio = AudioSegment.from_wav(wav_name)
        duration = math.floor(audio.duration_seconds)
        for i in range(0, int(duration // sec_per_split)):
            basename, ext = os.path.splitext(wav_name)
            split_fn = basename + "_" + str(i) + ext
            out_file = os.path.join(out_dir, os.path.basename(split_fn))
            split_audio = audio[i * 1000 : (i + sec_per_split) * 1000]
            # exclude samples less than 0.9 seconds
            if split_audio.duration_seconds > (sec_per_split - 0.01):
                split_audio.export(out_file, format="wav")


def _extract_feature(cfg, utt_id, feat_dir, is_train):
    """Perform feature extraction.

    Args:
        cfg (DictConfig): configuration in YAML format.
        utt_id (str): basename for audio.
        feat_dir (str): directory name for saving features.
        scaler (StandardScaler): standard scaler from scikit-learn.
        is_train (Bool): handling training dataset or test dataset.
    """
    if is_train is True:
        wav_dir = os.path.join(
            cfg.TOPR.root_dir,
            cfg.TOPR.data_dir,
            cfg.TOPR.trainset_dir,
            cfg.TOPR.split_dir,
        )
    else:
        wav_dir = os.path.join(
            cfg.TOPR.root_dir,
            cfg.TOPR.data_dir,
            cfg.TOPR.evalset_dir,
            cfg.TOPR.resample_dir,
        )
    wav_file = os.path.join(wav_dir, utt_id + ".wav")
    audio, rate = sf.read(wav_file)
    if audio.dtype in [np.int16, np.int32]:
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float64)
    audio = audio.astype(np.float64)

    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.feature.window, cfg.feature.win_length),
        hop=cfg.feature.hop_length,
        fs=rate,
        mfft=cfg.feature.n_fft,
    )
    stft_data = stfft.stft(audio)
    stft_data = stft_data.T  # transpose -> [n_frames, n_fft/2 +1]
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_logmag.npy"),
        np.log(np.abs(stft_data)).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_phase.npy"),
        np.angle(stft_data).astype(np.float32),
        allow_pickle=False,
    )


def extract_feature(cfg, is_train=True):
    """Extract acoustic features.

    Args:
        cfg (DictConfig): configuration in YAML format.
        is_train (Bool): handling training dataset or test dataset.
    """
    dataset_dir = cfg.TOPR.trainset_dir if is_train is True else cfg.TOPR.evalset_dir
    wav_dir = cfg.TOPR.split_dir if is_train is True else cfg.TOPR.resample_dir
    wav_list = os.listdir(
        os.path.join(cfg.TOPR.root_dir, cfg.TOPR.data_dir, dataset_dir, wav_dir)
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(
        cfg.TOPR.root_dir, cfg.TOPR.feat_dir, dataset_dir, cfg.feature.window
    )
    os.makedirs(feat_dir, exist_ok=True)

    with ProcessPoolExecutor(cfg.preprocess.n_jobs) as executor:
        futures = [
            executor.submit(_extract_feature, cfg, utt, feat_dir, is_train)
            for utt in utt_list
        ]
        for future in prg(
            futures,
            prefix="Extract acoustic features: ",
            suffix=" ",
            redirect_stdout=False,
        ):
            future.result()  # return None


def main(cfg):
    """Perform preprocess."""
    # training data
    get_basic5000_label(cfg)
    resample_wav(cfg)
    trim_silence(cfg)
    split_utterance(cfg)
    extract_feature(cfg)

    # test data
    resample_wav(cfg, is_train=False)
    extract_feature(cfg, is_train=False)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)
