import torch
from torch import nn
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model
import librosa
import numpy as np
from test_mel import get_mel
from transformers import AutoProcessor, Wav2Vec2ForCTC
from utils import load_wav_to_torch
from tacotron2.utils import get_mask_from_lengths

# from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# def get_FeatureExtractor():
#     feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
#     return feature_extractor


# def get_encoder():
#     model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#     model.dropout = Identity()
#     model.lm_head = Identity()
#     return model

def get_decoder():
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load('/home/bubee/accent_conversion/pretrained_model/tacotron2_statedict.pt')['state_dict'])
    return tacotron2.decoder
    
def get_postnet():
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load('/home/bubee/accent_conversion/pretrained_model/tacotron2_statedict.pt')['state_dict'])
    return tacotron2.postnet



def check_huggingface():
    audio, sample_rate = librosa.load('/home/bubee/accent_conversion/slt_native/arctic_a0001.wav', sr=22500)
    audio = torch.FloatTensor(audio.astype(np.float32))

    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    #memory: torch.Size([1, 235, 768])
    #decoder_inputs: torch.Size([1, 80, 294])
    #memory_lengths: torch.Size([2])
    input_values = feature_extractor(audio, return_tensors="pt").input_values
    model.dropout = Identity()
    model.lm_head = Identity()
    out = model(input_values).logits
    test_encoder_output = torch.FloatTensor(1, 235, 512)
    test_encoder_output = test_encoder_output.to('cuda:0', dtype=torch.float32)
    decoder = get_decoder()
    mel = get_mel('/home/bubee/accent_conversion/slt_native/arctic_a0001.wav')
    mel = mel[None, :, :]
    mel = mel.to('cuda:0', dtype=torch.float32)
    # mel = mel.transpose(2, 1)
    # memory(encoder output), decoder_inputs(GT mel for teacher forcing), memory_lengths(encoder output dim)
    out = out.to('cuda:0', dtype=torch.float32)
    mem_lengths = torch.IntTensor([40])
    mem_lengths = mem_lengths.to('cuda:0', dtype=torch.int)

    mel_outputs, gate_outputs, alignments = decoder(test_encoder_output, mel, mem_lengths)
    # print('model_dir', dir(model))
    # print('OK:', out)
    # print('dim:', out.size())

    # out = wav2vec2_feature_extractor(input_values)
    # out = wav2vec2_feature_projection(out)
    # out = wav2vec2_context_encoder(out)
    #
    # print('model(input_values):', out)

    print('ok:', mel_outputs)
    print('out_dim:', mel_outputs.size())

def checkforctcloss():
    audio, sample_rate = librosa.load('/home/bubee/accent_conversion/slt_native/arctic_a0002.wav', sr=16000)
    audio = torch.FloatTensor(audio.astype(np.float32))

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.dropout = Identity()
    model.lm_head = Identity()
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    logits = model(**inputs).logits

def get_feature_extractor():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    return processor

def get_wav2vec():
    model1 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model1.lm_head = Identity()
    return model1

def get_classifier():
    model2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    lm_head = nn.Linear(768, 32)
    lm_head.load_state_dict(model2.lm_head.state_dict())
    print('lm_head:', lm_head)
    return lm_head
    
    

class AccentConversion(nn.Module):
    
    def __init__(self):
        super(AccentConversion, self).__init__()
        # self.feature_extractor = get_feature_extractor()
        self.encoder = get_wav2vec()
        self.classifier = get_classifier()
        self.Linear = nn.Linear(768, 512)
        self.decoder = get_decoder()
        self.postnet = get_postnet()

        self.mask_padding = True
        self.n_mel_channels = 80

        
    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
            
        return outputs
        
    
    def forward(self, x, mel_GT, mem_length, processor, output_lengths):
        inputs = processor(x, sampling_rate=16000, return_tensors="pt")
        # print('size:', inputs['input_values'].size())
        inputs = inputs['input_values'].squeeze(dim=0)
        logits = self.encoder(inputs).logits
        # print('logits:', logits.size())
        ##############################################
        text = self.classifier(logits)
        # print('text:', text.size())
        # predicted_ids = torch.argmax(text, dim=-1)
        # print('predicted_ids:', predicted_ids.size())
        # transcription = processor.batch_decode(predicted_ids)
        # print('transcription:,', transcription)
        ##############################################
        encoder_output = self.Linear(logits)


        encoder_output = encoder_output.cuda(0)
        # print('encoder_output:', encoder_output.size())
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_output, mel_GT, memory_lengths=mem_length)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # print('mel_GT:', mel_GT.size())
        # print('mel_outputs:', mel_outputs.size())
        # print('mel_outputs_postnet:', mel_outputs_postnet.size())
        # return mel_outputs, mel_outputs_postnet, text
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths), text


# if __name__ == '__main__':
    # decoder = get_decoder()
    # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./pretrained_model/wav2vec_small.pt'])
    # hparams = create_hparams()
    # hparams.sampling_rate = 22050
    #
    # tacotron = load_model(hparams)
    # tacotron.load_state_dict(torch.load('/home/bubee/accent_conversion/pretrained_model/tacotron2_statedict.pt')['state_dict'])
    # print(dir(tacotron))

    # path = '/home/bubee/FreeVC/non_native_transform/asi_to_slt/arctic_a0001_target.wav'
    #
    # audio, sampling_rate = load_wav_to_torch(path)
    #
    # accent_model = AccentConversion()
    # wav_input_16khz = torch.randn(1, 10000)
    # accent_model(wav_input_16khz)

    # check_huggingface()
