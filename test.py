import torch
# from model import *
from data_loader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from mel_processing import spectrogram_torch, spec_to_mel_torch
from torch.nn import functional as F
from transformers import AutoProcessor, Wav2Vec2ForCTC

import numpy as np
from scipy.io.wavfile import write


import torch.nn as nn

###                                          test for save model and load model
# model = AccentConversion()
# torch.save(model.state_dict(), '/home/bubee/accent_conversion/ckpt.pt')
# model.load_state_dict(torch.load('/home/bubee/accent_conversion/ckpt.pt'))
# print('ok')



# def get_train_list():
#     path = '/home/bubee/FreeVC/non_native_transform/asi_to_slt/'
#     train_list = []
#     for file_name in os.listdir(path):
#         file_name = file_name.replace('_target.wav', '')
#         train_list.append(file_name)
#     print('train list already prepared.')
#     return train_list
#
# train_list = get_train_list()

def get_mel(spec):
    mel = spec_to_mel_torch(
        spec,
        1024,
        80,
        16000,
        0.0,
        None)
    return mel


def changeBack(arr):
    return [chr(num) for num in arr if num!=0]

###
train_dataset = TextAudioLoader()
collate_fn = TextAudioCollate()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, pin_memory=True, collate_fn=collate_fn)
# train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, collate_fn=collate_fn)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#audio_padded, audio_len, non_native_spec_padded, non_native_spec_len, native_spec_padded, native_spec_len, text_padded, text_len, text_length
for batch_idx, (a, a_lengths, spec, spec_lengths, s, s_lengths, t, t_len, t_length) in enumerate(train_loader):
    mel_GT = get_mel(s)
    a = a.cuda(0)
    mel_GT = mel_GT.cuda(0)
    t_length = t_length.cuda(0)
    t = t.cuda(0)
    t_len = t_len.cuda(0)
    # print('mel_GT:',mel_GT.size())
    # print('batch_idx : ', batch_idx)
    print('-------------------batch{}-------------------', batch_idx)
    # print('a.size()', a)
    # print('spec.size()', spec.size())
    # print('s.size()', s.size())
    # print('t.size()', t.size())
    # print('t_length', t_length)

    ### check during generator
    # inputs = processor(a, sampling_rate=16000, return_tensors="pt")
    # inputs = inputs['input_values'].squeeze(dim=0)
    # logits = model(inputs).logits
    # print(logits.size())

    # print('logits:', logits.size())
    # predicted_ids = torch.argmax(logits, dim=3)
    # print(predicted_ids.size())
    # transcription = processor.batch_decode(predicted_ids)
    # print(transcription)


    ### check if audio is correct
    # if(batch_idx==0):
    #     a = a.cpu()
    #     new_data = a[1].numpy()
    #     # scaled = np.int16(new_data / np.max(np.abs(new_data)) * 32767)
    #     # write('test3.wav', 16000, scaled)
    #     print(new_data)
    #     break

    ###test ctc loss
    # if(batch_idx==0):
    #     target_list=[]
    #     for i in range(64):
    #         x = t[i]
    #         target_list+=changeBack(x)
    #     word = ''.join(symbol for symbol in target_list)
        
    #     # print(word)
    #     target = processor(text=word, return_tensors="pt").input_ids
    #     target = target.squeeze(0)
    #     print('target:', target.size())
    #     print('t_len:', sum(t_len))
    #     # temp = torch.randint([64, 296, 32])
    #     input = torch.transpose(logits, 0, 1)
        
    #     ctc_loss = nn.CTCLoss()
    #     loss = ctc_loss(input, target, t_length, t_len)
    #     print(loss)
        
    #     break
        
    a, a_lengths, spec, spec_lengths, s, s_lengths, t, t_len, t_length
    ### check if tensors have nan
    if torch.isnan(a).any(): print('got you')
    if torch.isnan(a_lengths).any(): print('got you')
    if torch.isnan(spec).any(): print('got you')
    if torch.isnan(spec_lengths).any(): print('got you')
    if torch.isnan(s).any(): print('got you')
    if torch.isnan(s_lengths).any(): print('got you')
    if torch.isnan(t).any(): print('got you')
    if torch.isnan(t_len).any(): print('got you')
    if torch.isnan(t_length).any(): print('got you')
print('all are fine')
    # accent_model = AccentConversion()
    # mel_outputs, mel_outputs_postnet = accent_model(a, mel_GT, t_length)
    # loss = F.l1_loss(mel_GT, mel_outputs)
    # print('loss:', loss)
    


# text, source_spec, target_spec = train_dataset[0]

# train = TextAudioLoader()
# train.get_text_audio_target('arctic_a0001', '/home/bubee/FreeVC/non_native_transform/asi_to_slt/')