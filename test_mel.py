import torch
import os
import torchaudio
from torchaudio import transforms
import numpy as np
import sys
import numpy
import struct
import warnings
from enum import IntEnum
from scipy.io.wavfile import read
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text
import librosa
import sys


max_wav_value=32768.0

def get_audio(file_path_and_name):
    audio, sample_rate = librosa.load(file_path_and_name, sr=22500)
    audio = torch.FloatTensor(audio.astype(np.float32))
    # audio, sampling_rate = load_wav_to_torch(file_path_and_name)
    print('audio.size():',audio.size())
    audio_norm = audio / max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, 1024,
                             22050, 256, 1024,
                             center=False)
    spec = torch.squeeze(spec, 0)
    return spec
    # waveform, sample_rate = torchaudio.load(file_path_and_name, normalize=True)
    # transform = transforms.MelSpectrogram(sample_rate)
    # mel_specgram = transform(waveform)
    # return mel_specgram

# mel = get_audio('/home/bubee/Downloads/non_native_speech/l2arctic_release_v5.0/ASI/wav/arctic_a0001.wav')
# mel = get_audio('/home/bubee/FreeVC/non_native_transform/asi_to_slt/arctic_a0001_target.wav')
# print(mel.size())


# "mel_fmin": 0.0
# "mel_fmax": null
# mel = spec_to_mel_torch(
#           mel,
#           1024,
#           80,
#           16000,
#           0.0,
#           None)
# print('mel.size():', mel.size())


def get_mel(autio_path):
    spec = get_audio(autio_path)
    mel = spec_to_mel_torch(
        spec,
        1024,
        80,
        22050,
        0.0,
        None)
    return mel


# print('finally:', get_mel('/home/bubee/accent_conversion/slt_native/arctic_a0001.wav'))


