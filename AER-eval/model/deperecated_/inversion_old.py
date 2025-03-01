import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
import torchcrepe
from scipy.signal import butter, lfilter, filtfilt, resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel


def butter_bandpass(cut, fs, order=5):
    
    if isinstance(cut,list) and len(cut) == 2:
        return butter(order, cut, fs=fs, btype='bandpass')
    else:
        return butter(order, cut, fs=fs, btype='low')

def butter_bandpass_filter(data, cut, fs, axis=1, order=5):
    b, a = butter_bandpass(cut, fs, order=order)
    y = filtfilt(b, a, data,axis=axis)
    return y

class AmplitudeHistogram(torch.nn.Module):
    
    def __init__(self, hop_length):
        super().__init__()
        kernel = torch.ones(hop_length)/hop_length
        kernel = kernel.unsqueeze(0)
        self.conv = torch.nn.Conv1d(1, 1, hop_length, stride=hop_length, padding=hop_length//2,)
        self.conv.weight.data = kernel.unsqueeze(1)
        self.conv.requires_grad_(False)
        
    def forward(self, x):
        return self.conv(x.unsqueeze(1).abs()).squeeze(1)

class SpeechWave(object):
    def __init__(self, input_values, input_lens):
        self.input_values = input_values
        self.input_lens = input_lens
        
    def to(self, device):
        self.input_values=self.input_values.to(device)
        return self
        
        

class Inversion():
    
    def __init__(self, linear_model_path, speech_model='facebook/wav2vec2-large-xlsr-53', target_layer=17, freqcut=10,
                 device='cuda', normalize=True,pitch_q=1 ):
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        if 'wavlm' in speech_model:
            self.speech_model = WavLMModel.from_pretrained(speech_model)
        else:
            self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
        self.speech_model = self.speech_model.eval().to(device)
        self.sr = 16000
        self.linear_model = pickle.load(open(linear_model_path,'rb'))
        self.tgt_layer = target_layer
        self.freqcut = freqcut
        self.ft_sr = 50
        self.fmin = 50
        self.fmax = 550
        self.q = pitch_q
        self.sr = 16000
        self.pitch_hop_length = int(self.sr/(self.ft_sr*self.q))
        self.intensity_hop_length = int(self.sr/self.ft_sr)
        self.device = device
        self.crepe_model = 'full'
        self.ssl_states_ = None
        self.normalize= normalize
        self.intensity_model = AmplitudeHistogram(self.intensity_hop_length)
        self.intensity_model = self.intensity_model.eval().to(device)
        
    def _extract_ema(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            outputs = self.speech_model(wavs.input_values,
                                        output_hidden_states=True)
        states=outputs.hidden_states
        low_acoustics_ = states[0].cpu().numpy()
        states=states[self.tgt_layer].cpu().numpy()
        if self.freqcut>0:
            states=butter_bandpass_filter(states,self.freqcut,self.ft_sr,axis=1)
        state_shape = states.shape
        ema = self.linear_model.predict(states.reshape(-1,state_shape[-1])).reshape(state_shape[0],state_shape[1],12)
        
        outputs["wav"] = wavs
        outputs["acoustics"] = low_acoustics_
        outputs["ema"] = ema
        return outputs
        
    
    def _extract_pitch(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            pitch,confidence = torchcrepe.predict(wavs.input_values,
                                                   self.sr,
                                                   self.pitch_hop_length,
                                                   self.fmin,
                                                   self.fmax,
                                                   self.crepe_model,
                                                   batch_size=2048,
                                                   device=self.device,
                                                   return_periodicity=True)
       
        pitch = (pitch-self.fmin)/(self.fmax-self.fmin)
        
        def _reshape(arr,q):
            b = arr.shape[0]
            l = arr.shape[1]
            arr = arr[:,:int(l//q)*q]
            arr = arr.reshape(b,l//q,q)
            arr = arr.mean(-1)
            return arr
       
        pitch = _reshape(pitch, self.q).cpu().numpy()
        confidence = _reshape(confidence, self.q).cpu().numpy()
        outputs["wav"]=wavs
        outputs["pitch"] = pitch
        outputs["confidence"] = confidence
        return outputs
        
    def _extract_intensity(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            intensity = self.intensity_model(wavs.input_values).cpu().numpy()
        outputs["wav"] = wavs
        outputs["intensity"] = intensity
        return outputs
    
    def _match_and_cat(self, arrs):
        min_len = np.min([len(arr) for arr in arrs])
        arrs = [arr[:min_len] for arr in arrs]
        arrs = [arr[:,None] if len(arr.shape)==1 else arr for arr in arrs ]
        arrs = np.concatenate(arrs,-1)
        return arrs
    
    def _normalize_pitch(self, pitch, confidence):
        weighted_mean = (pitch*confidence).sum()/confidence.sum()
        weighted_var = (((pitch-weighted_mean)**2)*confidence).sum()/confidence.sum()
        weighted_std = weighted_var**.5
        normalized_pitch = (pitch-weighted_mean)/weighted_std
        return normalized_pitch, weighted_mean, weighted_std
    
    def _get_spk_emb(self, acoustics, weights):
        min_len_ = min(len(acoustics), len(weights))
        acoustics = acoustics[:min_len_]
        weights = weights[:min_len_][:,None]
        return (acoustics*weights).sum(0)/weights.sum()
    
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = int(len_/self.sr*self.ft_sr)
            ema = outputs['ema'][b]
            intensity = outputs["intensity"][b]
            pitch, pitch_mean, pitch_std = self._normalize_pitch(outputs['pitch'][b],
                                                                 outputs['confidence'][b])
            art = self._match_and_cat([ema, pitch, intensity])
            spk_emb = self._get_spk_emb(outputs['acoustics'][b],
                                        outputs['confidence'][b])
            split_outputs.append({"art":art,
                                   "spk_emb":spk_emb,
                                   "pitch_stats":np.array([pitch_mean, pitch_std])})
        return split_outputs
    
    def _load_wav(self, wav):
        wav,sr = sf.read(wav)
        if len(wav.shape)>1:
            wav = wav.mean(-1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        if self.normalize:
            wav = (wav-wav.mean())/wav.std()
        return wav
    
    def process_wavfiles(self, wavfiles):
        if isinstance(wavfiles, str) or isinstance(wavfiles, Path):
            wavfiles = [wavfiles]
        wavs = [self._load_wav(wavfile) for wavfile in wavfiles]
        wavs = [torch.from_numpy(wav).float() for wav in wavs]
        input_lens = [len(wav) for wav in wavs]
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        wavs = SpeechWave(input_values=wavs, input_lens=input_lens)
        wavs = wavs.to(self.device)
        return wavs
    
    def __call__(self, wavfiles):
        wavs = self.process_wavfiles(wavfiles)
        outputs = {}
        outputs = self._extract_ema(wavs, outputs)
        outputs = self._extract_pitch(wavs, outputs)
        outputs = self._extract_intensity(wavs, outputs)
        outputs = self._split_batch(outputs)
        return outputs

class ExtractAvgSpkEmb():
    
    def __init__(self, speech_model='facebook/wav2vec2-large-xlsr-53',device='cuda', normalize=True):
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        if 'wavlm' in speech_model:
            self.speech_model = WavLMModel.from_pretrained(speech_model)
        else:
            self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
        self.speech_model = self.speech_model.eval().to(device)
        self.sr = 16000
        self.device = device
        self.normalize= normalize
        self.ft_sr = 50
        
    def _extract_fts(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            outputs = self.speech_model(wavs.input_values,
                                        output_hidden_states=True)
        states=outputs.hidden_states
        low_acoustics_ = states[0].cpu().numpy()
        outputs["acoustics"] = low_acoustics_
        outputs["wav"] = wavs
        return outputs
        
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = int(len_/self.sr*self.ft_sr)
            
            spk_emb = outputs['acoustics'][b][:ft_len].mean(0)
            split_outputs.append({"spk_emb":spk_emb,})
        return split_outputs
    
    def _load_wav(self, wav):
        wav,sr = sf.read(wav)
        if len(wav.shape)>1:
            wav = wav.mean(-1)
        if sr != self.sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        if self.normalize:
            wav = (wav-wav.mean())/wav.std()
        return wav
    
    def process_wavfiles(self, wavfiles):
        if isinstance(wavfiles, str) or isinstance(wavfiles, Path):
            wavfiles = [wavfiles]
        wavs = [self._load_wav(wavfile) for wavfile in wavfiles]
        wavs = [torch.from_numpy(wav).float() for wav in wavs]
        input_lens = [len(wav) for wav in wavs]
        wavs = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
        wavs = SpeechWave(input_values=wavs, input_lens=input_lens)
        wavs = wavs.to(self.device)
        return wavs
    
    def __call__(self, wavfiles):
        wavs = self.process_wavfiles(wavfiles)
        outputs = {}
        outputs = self._extract_fts(wavs, outputs)
        outputs = self._split_batch(outputs)
        return outputs
            
