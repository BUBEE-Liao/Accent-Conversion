import torch
from data_loader import *
from mel_processing import spectrogram_torch, spec_to_mel_torch
from model import *
from torch.nn import functional as F
import os
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from torch.utils.tensorboard import SummaryWriter
from hifigan.env import AttrDict
from hifigan.models import Generator
import json
#test main#
#yoro#

def main():

    ##### accent conversion model #####
    model = AccentConversion()
    
    ##### optimizer #####
    learning_rate=5e-6 #0.5 in original paper
    weight_decay=1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    ##### data lodaer #####
    batch_size=64
    train_dataset = TextAudioLoader()
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    ##### start training #####
    train(model, optimizer, train_loader)


def train(model, optimizer, train_loader):
    #########for evaluate
    learning_rate=5e-6
    hifi_gan = load_hifigan()
    writer = SummaryWriter(log_dir='./logs')
    #########
    batch_size = 64
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    accent_model = model
    accent_model.train()
    accent_model.encoder.wav2vec2.feature_extractor.eval()
    accent_model.classifier.eval()

    ctc_loss = nn.CTCLoss()
    l2_loss = nn.MSELoss()
    
    iteration = 1
    epochs = 1000000 #where to stop training
    print_iteration = 1
    eval_iteration = 30
    store_iteration = 300
    for epoch in range(0, epochs):
        print('------epoch{}------'.format(epoch))
        for batch_idx, (audio, audio_len, nonnative_mel_spec, nonnative_mel_spec_len, native_mel_spce, native_mel_spec_len, text, text_len, memory_length) in enumerate(train_loader):
            mel_GT = native_mel_spce
            audio = audio.cuda(0)
            mel_GT = mel_GT.cuda(0)
            memory_length = memory_length.cuda(0)
            native_mel_spec_len = native_mel_spec_len.cuda(0)

            # inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            # inputs = inputs['input_values'].squeeze(dim=0)
            # inputs.cuda(0)
            
            #forward
            # accent_model.zero_grad()
            model_output, phoneme_predicted = accent_model(audio, mel_GT, memory_length, processor, native_mel_spec_len)
            mel_out, mel_out_postnet, gate_out, _ = model_output

            #process text
            #calculate ctc loss
            target_list=[]
            for i in range(list(text_len.size())[0]):
                x = text[i]
                target_list += [chr(num) for num in x if num!=0]
            word = ''.join(symbol for symbol in target_list)

            target = processor(text=word, return_tensors="pt").input_ids
            target = target.squeeze(0)

            input = torch.transpose(phoneme_predicted, 0, 1)
            input = input.log_softmax(2)

            loss_ctc = ctc_loss(input, target, memory_length, text_len)
            
            # calculate loss
            # decoder_mel_loss = F.l2_loss(mel_GT, decoder_output_mel)
            # postnet_mel_loss = F.l2_loss(mel_GT, postnet_output_mel)

            decoder_mel_loss = l2_loss(mel_GT, mel_out)
            postnet_mel_loss = l2_loss(mel_GT, mel_out_postnet)

            if(iteration%print_iteration==0):
                print('iteration{}: '.format(iteration), 'decoder_mel_loss:', decoder_mel_loss.item(), ', postnet_mel_loss:', postnet_mel_loss.item(), 'ctc_loss:', loss_ctc.item())

            if(iteration%eval_iteration==0):
                evaluate(mel_out_postnet[0, :, :], mel_GT[0, :, :], hifi_gan, writer, iteration)

            #####calculate total loss
             
                
            total_loss = decoder_mel_loss + postnet_mel_loss + loss_ctc

            #print loss
            optimizer.zero_grad()
            
            # if(iteration%eval_iteration==0):
            #     print('iteration{}: '.format(iteration), 'decoder_mel_loss:', decoder_mel_loss.item(), ', postnet_mel_loss:', postnet_mel_loss.item())

            #backward
            total_loss.backward()
            
            #gradient clip
            torch.nn.utils.clip_grad_norm_(parameters=accent_model.parameters(), max_norm=0.02208, norm_type=2.0)
            
            optimizer.step()



            if(iteration%store_iteration==0):
                ckpt_path = os.path.join('/home/bubee/accent_conversion_git/test_change/ckpt/', "iteration_{}.pth".format(iteration))
                state_dict = accent_model.state_dict()
                torch.save({'model': state_dict,'iteration': iteration,'optimizer': optimizer.state_dict(),'learning_rate': learning_rate}, ckpt_path)

                
            iteration += 1
                                         

def evaluate(mel_out_postnet, mel_GT, hifi_gan, writer, step):
    mel_out_postnet = mel_out_postnet.unsqueeze(0)
    mel_out_postnet = mel_out_postnet.cuda(0)
    y_g_hat = hifi_gan(mel_out_postnet)
    y_gt = hifi_gan(mel_GT)
    
    audio_hat = y_g_hat.squeeze()
    audio_gt = y_gt.squeeze()
    
    audio_hat = audio_hat.cpu().detach().numpy().astype('float32')
    audio_gt = audio_gt.cpu().detach().numpy().astype('float32')
    
    writer.add_audio('audio_hat_{}'.format(step), audio_hat, global_step=step, sample_rate=22050)
    writer.add_audio('audio_gt_{}'.format(step), audio_gt, global_step=step, sample_rate=22050)
    
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def load_hifigan():
    state_dic = '/home/bubee/accent_conversion_git/test_change/hifigan/VCTK_V1-20230803T150056Z-001/VCTK_V1/'
    config_file = os.path.join(state_dic, 'config.json')
    
    with open(config_file) as f:
        data = f.read()
        
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(state_dic+'generator_v1', device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    return generator

def get_mel(spec):
    sr = 22050
    mel = spec_to_mel_torch(
        spec,
        1024,
        80,
        sr,
        0.0,
        8000)
    return mel

def turnTextIntoNum(transcription):
    return None



if __name__ == '__main__':
    main()
