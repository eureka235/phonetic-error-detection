import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
import torchcrepe
#import penn
from .speech import BaseExtractor, SpeechWave

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
        
        
class SourceExtractor(BaseExtractor):
    
    def __init__(self, device='cuda', normalize=True,pitch_q=1, ft_sr=50, fmin=50,
                fmax=550, sr=16000, crepe_model="full"):
        
        self.ft_sr = ft_sr
        self.fmin = fmin
        self.fmax = fmax
        self.q = pitch_q
        self.sr = sr
        self.pitch_hop_length = int(self.sr/(self.ft_sr*self.q))
        self.intensity_hop_length = int(self.sr/self.ft_sr)
        self.device = device
        self.crepe_model = crepe_model
        self.normalize= normalize
        self.intensity_model = AmplitudeHistogram(self.intensity_hop_length)
        self.intensity_model = self.intensity_model.eval().to(device)
        
    def _extract_pitch(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            pitch, periodicity = torchcrepe.predict(wavs.input_values,
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
        periodicity = _reshape(periodicity, self.q).cpu().numpy()
        outputs["wav"]=wavs
        outputs["pitch"] = pitch
        outputs["periodicity"] = periodicity
        return outputs
        
    def _extract_intensity(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            intensity = self.intensity_model(wavs.input_values).cpu().numpy()
        outputs["wav"] = wavs
        outputs["loudness"] = intensity
        return outputs
    
    def _normalize_pitch(self, pitch, confidence):
        weighted_mean = (pitch*confidence).sum()/confidence.sum()
        weighted_var = (((pitch-weighted_mean)**2)*confidence).sum()/confidence.sum()
        weighted_std = weighted_var**.5
        normalized_pitch = (pitch-weighted_mean)/weighted_std
        return normalized_pitch, weighted_mean, weighted_std
    
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = int(len_/self.sr*self.ft_sr)
            intensity = outputs["intensity"][b]
            periodicity = outputs["periodicity"][b]
            pitch = outputs['pitch'][b]
            normalized_pitch, pitch_mean, pitch_std = self._normalize_pitch(outputs['pitch'][b],
                                                                 outputs['periodicity'][b])
            art = self._match_and_cat([pitch, intensity,periodicity,normalized_pitch])
            split_outputs.append({"pitch":art[:,0],
                                  "loudness": art[:,1],
                                  "periodicity":art[:,2],
                                  "normalized_pitch": art[:,3],
                                  "pitch_stats":np.array([pitch_mean, pitch_std])})
        return split_outputs
    
    
    def __call__(self, wavfiles):
        wavs = self.process_wavfiles(wavfiles)
        outputs = {}
        outputs = self._extract_pitch(wavs, outputs)
        outputs = self._extract_intensity(wavs, outputs)
        outputs = self._split_batch(outputs)
        return outputs
    
    