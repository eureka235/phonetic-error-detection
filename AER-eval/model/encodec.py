import torch
import numpy as np
from .speech import BaseExtractor, SpeechWave
from .inversion import Inversion
from .src_extractor import SourceExtractor
from .spk_encoder import SpeakerEncoder
from .generator import HiFiGANGenerator

def hz2mel(x, method="htk"):
    if method=="htk":
        mel = 2595*np.log10(1+x/700)
    elif method=="none":
        mel = x
    else: #regular Slaney's Mel which is linear under 1000
        mel = x*3/200
    return mel

def mel2hz(mel, method="htk"):
    if method=="htk":
        hz = (np.power(10, mel/2595)-1)*700
    elif method=="none":
        hz = mel
    else:
        hz = mel*200/3
    return hz

class ArticulatoryEncodec(BaseExtractor):
    def __init__(self, linear_model_path, spk_ft_ckpt, generator_ckpt,
                 generator_configs, 
                 speech_model='facebook/wav2vec2-large-xlsr-53', 
                 target_layer=17, freqcut=10, pitch_q=1, fmin=50, fmax=550,
                 crepe_model="full", spk_ft_size=1024, spk_emb_size=64,
                 device='cuda', normalize=True, sr=16000, ft_sr=50, pitch_scale=5,
                 pitch_stats_method="htk"):
        
        common_configs = {"device":device, "normalize":normalize, "sr":sr,
                          "ft_sr":ft_sr}
        self.inverter = Inversion(linear_model_path, speech_model=speech_model,
                                  target_layer=target_layer, freqcut=freqcut, **common_configs)
        self.source_extractor =  SourceExtractor(pitch_q=pitch_q, fmin=fmin,
                                                 fmax=fmax, crepe_model=crepe_model,
                                                 **common_configs)
        self.speaker_encoder = SpeakerEncoder(spk_ft_ckpt=spk_ft_ckpt, spk_ft_size=spk_ft_size,
                                              spk_emb_size=spk_emb_size,
                                              speech_model=None, **common_configs)
        
        self.device = device
        generator_configs["spk_emb_size"] = spk_emb_size
        self.generator = HiFiGANGenerator(**generator_configs)
        self.generator.load_state_dict(torch.load(generator_ckpt, map_location="cpu"))
        self.generator.remove_weight_norm()
        self.generator = self.generator.eval().to(self.device)
        self.sr = sr
        self.ft_sr = ft_sr
        self.normalize = normalize
        self.pitch_scale = pitch_scale
        self.pitch_stats_method = pitch_stats_method
    
    def _get_pitch_stats(self, pitch, confidence):
        fmin = self.source_extractor.fmin
        fmax = self.source_extractor.fmax
        pitch = (pitch*(fmax-fmin)+fmin)
        pitch = hz2mel(pitch, self.pitch_stats_method)
        weighted_mean = (pitch*confidence).sum()/confidence.sum()
        weighted_var = (((pitch-weighted_mean)**2)*confidence).sum()/confidence.sum()
        weighted_std = weighted_var**.5
        
        return weighted_mean, weighted_std
    
    def _split_batch(self, outputs):
        batch_size = outputs["wav"].input_values.shape[0]
        split_outputs = []
        for b in range(batch_size):
            len_ = outputs["wav"].input_lens[b]
            ft_len = int(len_/self.sr*self.ft_sr)
            periodicity = outputs["periodicity"][b]
            acoustics = outputs["acoustics"][b]
            loudness = outputs["loudness"][b]
            pitch = outputs["pitch"][b]
            ema = outputs["ema"][b]
            if batch_size>1:
                periodicity = periodicity[:ft_len]
                acoustics = acoustics[:ft_len]
                pitch = pitch[:ft_len]
                ema = ema[:ft_len]
                loudness = loudness[:ft_len]
            spk_emb = self.speaker_encoder._get_spk_emb(acoustics, periodicity)
            spk_emb = torch.from_numpy(spk_emb).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                spk_emb = self.speaker_encoder.spk_enc(spk_emb)
            spk_emb = spk_emb.squeeze(0).cpu().numpy()
            art = self._match_and_cat([ema, pitch, loudness])
            pitch_mean, pitch_std = self._get_pitch_stats(pitch, periodicity)
            split_outputs.append({"art":art, "spk_emb": spk_emb, "pitch_stats":[pitch_mean, pitch_std]})
        return split_outputs
    
    def encode(self, wav_file):
        wavs = self.process_wavfiles(wav_file)
        outputs = {}
        outputs = self.inverter._extract_ema(wavs, outputs, include_acoustics=True)
        outputs = self.source_extractor._extract_pitch(wavs, outputs)
        outputs = self.source_extractor._extract_intensity(wavs, outputs)
        outputs = self._split_batch(outputs)[0]
        return outputs
    
    def decode(self, art, spk_emb, **kwargs):
        art = torch.from_numpy(art).float().T.to(self.device).unsqueeze(0)
        art[:,12,:] = art[:,12,:]*self.pitch_scale
        spk_emb = torch.from_numpy(spk_emb).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            cout = self.generator(art, spk_emb)
        wav = cout[0][0].cpu().numpy()
        return wav
        
    
    def _shift_pitch(self, pitch, original_stats, target_stats):
        
        fmin = self.source_extractor.fmin
        fmax = self.source_extractor.fmax
        pitch = (pitch*(fmax-fmin)+fmin)
        pitch = hz2mel(pitch, self.pitch_stats_method)
        pitch = (pitch-original_stats[0])/original_stats[1]
        pitch = pitch*target_stats[1]+target_stats[0]
        pitch = mel2hz(pitch, self.pitch_stats_method)
        pitch = (pitch-fmin)/(fmax-fmin)
        return pitch
        
    def convert(self, src_wav_file, trg_wav_file):
        src_code = self.encode(src_wav_file)
        trg_code = self.encode(trg_wav_file)
        src_code['art'][:,12] = self._shift_pitch(src_code['art'][:,12],
                                                  src_code["pitch_stats"],
                                                  trg_code["pitch_stats"])
        src_code["spk_emb"] = trg_code["spk_emb"]
        wav = self.decode(**src_code)
        return wav
        

