import torch
import numpy as np
import tqdm
import pickle
from scipy.signal import butter, lfilter, filtfilt, resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel
from .speech import BaseExtractor, SpeechWave

def butter_bandpass(cut, fs, order=5):
    
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, axis=1, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y

class Inversion(BaseExtractor):
    
    def __init__(self, linear_model_path, speech_model='facebook/wav2vec2-large-xlsr-53', target_layer=17, freqcut=10,
                 device='cuda', normalize=True, sr=16000, ft_sr=50):
    
        if 'wavlm' in speech_model:
            self.speech_model = WavLMModel.from_pretrained(speech_model)
        else:
            self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
        self.speech_model.encoder.layers = self.speech_model.encoder.layers[:target_layer+1]
        self.speech_model = self.speech_model.eval().to(device)
        self.sr = sr
        self.linear_model = pickle.load(open(linear_model_path,'rb'))
        self.tgt_layer = target_layer
        self.ft_sr = ft_sr
        self.device = device
        self.freqcut = freqcut
        self.normalize= normalize
        
    def _extract_ema(self, wavs, outputs={}, include_acoustics=False):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            speech_outputs = self.speech_model(wavs.input_values,
                                        output_hidden_states=True)
        states=speech_outputs.hidden_states
        states=states[self.tgt_layer].cpu().numpy()
        if self.freqcut>0:
            states=butter_bandpass_filter(states,self.freqcut,self.ft_sr,axis=1)
        state_shape = states.shape
        # print("state_shape: ", state_shape)
        ema = self.linear_model.predict(states.reshape(-1,state_shape[-1])).reshape(state_shape[0],state_shape[1],12)
        
        outputs["wav"] = wavs
        outputs["ema"] = ema
        if include_acoustics:
            low_acoustics_ = speech_outputs.hidden_states[0].cpu().numpy()
            outputs["acoustics"] = low_acoustics_
        return outputs
        
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = int(len_/self.sr*self.ft_sr)
            ema = outputs['ema'][b]
            if batch_size >1 :
                ema = ema[:ft_len]
            art = ema
            split_outputs.append({"art":art})
        return split_outputs
    
    def __call__(self, wavfiles):
        wavs = self.process_wavfiles(wavfiles)
        outputs = {}
        outputs = self._extract_ema(wavs, outputs)
        outputs = self._split_batch(outputs)
        return outputs
