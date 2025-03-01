import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa

class SpeechWave(object):
    def __init__(self, input_values, input_lens):
        self.input_values = input_values
        self.input_lens = input_lens
        
    def to(self, device):
        self.input_values=self.input_values.to(device)
        return self

class BaseExtractor(object):
    
    def __init__(self,sr=16000, ft_sr=50, normalize=True, device="cuda"):
        self.sr = sr
        self.ft_sr = ft_sr
        self.normalize= normalize
        self.device = device

    def _match_and_cat(self, arrs):
        min_len = np.min([len(arr) for arr in arrs])
        arrs = [arr[:min_len] for arr in arrs]
        arrs = [arr[:,None] if len(arr.shape)==1 else arr for arr in arrs ]
        arrs = np.concatenate(arrs,-1)
        return arrs
    
    def _split_batch(self, outputs):
        pass
    
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
        pass