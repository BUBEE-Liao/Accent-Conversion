import torch
import os
import torchaudio
from torchaudio import transforms
import librosa
import numpy as np
import sys
import numpy
import struct
import warnings
from enum import IntEnum
from scipy.io.wavfile import read
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text
import math
import torchaudio.functional as F_audio



class TextAudioLoader(torch.utils.data.Dataset):
# class TextAudioLoader():
    def __init__(self):
        super(TextAudioLoader, self).__init__()
        self.native_wav_path='/home/bubee/data/nas05/bubee/accent_conversion/slt_native_wav/'
        self.non_native_wav_path='/home/bubee/data/nas05/bubee/accent_conversion/slt_nonnative_wav/'
        self.text_path = '/home/bubee/accent_conversion/transcript/'
        self.train_list = self.get_train_list()


    ### PARA from VITS
        self.filter_length=1024
        self.sampling_rate = 16000
        self.hop_length = 256
        self.win_length = 1024
        self.max_wav_value = 32768.0
    def get_train_list(self):
        path = '/home/bubee/data/nas05/bubee/accent_conversion/slt_nonnative_wav/'
        train_list = []
        for file_name in os.listdir(path):
            file_name = file_name.replace('_target.wav', '')
            train_list.append(file_name)
        # print('train list already prepared.')
        return train_list


    def get_text(self, file_name):
        file_path = self.text_path+file_name+'.txt'
        with open(file_path, 'r')as f:
            line = f.readline()
            line = line.upper()
            alpha_list = [ord(char) for char in line]
        return alpha_list

    def get_audio(self, file_path_and_name):
        audio, sample_rate = librosa.load(file_path_and_name, sr=self.sampling_rate)
        audio = torch.FloatTensor(audio.astype(np.float32))
        audio = F_audio.resample(audio, self.sampling_rate, 22050)
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                22050, self.hop_length, self.win_length,
                                center=False)
        spec = torch.squeeze(spec, 0)
        return self.get_mel(spec)
        # audio, sampling_rate = load_wav_to_torch(file_path_and_name)
        # spec = spectrogram_torch(audio, self.filter_length,
        #                          self.sampling_rate, self.hop_length, self.win_length,
        #                          center=False)
        # spec = torch.squeeze(spec, 0)
        # return spec
        # waveform, sample_rate = torchaudio.load(file_path_and_name, normalize=True)
        # transform = transforms.MelSpectrogram(sample_rate)
        # mel_specgram = transform(waveform)
        # return mel_specgram

    def get_text_audio_target(self, file_name, non_native_wav_path, native_wav_path):  ###### remember to add audio
        source_path = non_native_wav_path+file_name+'_target.wav'
        target_path = native_wav_path+file_name+'.wav'
        
        source_spec = self.get_audio(source_path)
        target_spec = self.get_audio(target_path)
        
        source_audio, sample_rate = librosa.load(source_path, sr=self.sampling_rate)
        source_audio = torch.FloatTensor(source_audio.astype(np.float32))
        
        text_length = self.cal_dim(source_audio.size(dim=0))
        text_length = torch.IntTensor([text_length])

        text = self.get_text(file_name)
        text = torch.IntTensor(text)
        # print('source_spec:', source_spec)
        # print('source_spec:', source_spec.size())
        # print('target_spec:', target_spec)
        # print('target_spec:', target_spec.size())
        return source_audio, source_spec, target_spec, text, text_length

    def cal_dim(self, input_len):
        input_len-=1
        input_len/=5
        input_len+=1
        input_len = math.floor(input_len)
    
        for _ in range(0, 6):
            input_len-=1
            input_len/=2
            input_len+=1
            input_len = math.floor(input_len)
        return input_len-1

    def get_mel(self, spec):
        sr = 22050
        mel = spec_to_mel_torch(
            spec,
            1024,
            80,
            sr,
            0.0,
            8000)
        return mel


    def __getitem__(self, index):
        return self.get_text_audio_target(self.train_list[index], self.non_native_wav_path, self.native_wav_path)

    def __len__(self):
        return len(self.train_list)

class TextAudioCollate():

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)
        # print('ids_sorted_decreasing:',ids_sorted_decreasing)


        max_audio_len = max([len(x[0]) for x in batch])
        max_non_native_mel_spec_len = max([x[1].size(1) for x in batch])
        max_native_mel_spec_len = max([x[2].size(1) for x in batch])
        max_text_len = max([len(x[3]) for x in batch])

        audio_len = torch.LongTensor(len(batch))
        non_native_mel_spec_len = torch.LongTensor(len(batch))
        native_mel_spec_len = torch.LongTensor(len(batch))
        text_len = torch.LongTensor(len(batch))

        audio_padded = torch.FloatTensor(len(batch), max_audio_len)
        non_native_mel_spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_non_native_mel_spec_len)
        native_mel_spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_native_mel_spec_len)
        text_padded = torch.IntTensor(len(batch), max_text_len)
        text_length_batch = torch.IntTensor(len(batch))
        
        
        audio_padded.zero_()
        non_native_mel_spec_padded.zero_()
        native_mel_spec_padded.zero_()
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            audio = row[0]
            audio_padded[i, :audio.size(0)] = audio
            audio_len[i] = audio.size(0)

            non_native_mel_spec = row[1]
            non_native_mel_spec_padded[i, :, :non_native_mel_spec.size(1)] = non_native_mel_spec
            non_native_mel_spec_len[i] = non_native_mel_spec.size(1)

            native_mel_spec = row[2]
            native_mel_spec_padded[i, :, :native_mel_spec.size(1)] = native_mel_spec
            native_mel_spec_len[i] = native_mel_spec.size(1)

            text = row[3]
            text_padded[i, :text.size(0)] = text
            text_len[i] = text.size(0)

            text_length = row[4]
            text_length_batch[i] = text_length

        if self.return_ids: #text_padded, text_lengths,
            return audio_padded, audio_len, non_native_mel_spec_padded, non_native_mel_spec_len, native_mel_spec_padded, native_mel_spec_len, text_padded, text_len, text_length_batch, ids_sorted_decreasing
        return audio_padded, audio_len, non_native_mel_spec_padded, non_native_mel_spec_len, native_mel_spec_padded, native_mel_spec_len, text_padded, text_len, text_length_batch


