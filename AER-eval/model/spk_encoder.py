import torch
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import pickle
import librosa
from .speech import BaseExtractor, SpeechWave
from .src_extractor import SourceExtractor
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel

class SpeakerEncodingLayer(torch.nn.Module):
    def __init__(self, spk_ft_size=1024, spk_emb_size=64):
        super().__init__()
        self.spk_fc = torch.nn.Sequential(torch.nn.Linear(spk_ft_size, spk_ft_size),
                                          torch.nn.GELU(),
                                          torch.nn.Dropout(0.2),
                                          torch.nn.Linear(spk_ft_size, spk_emb_size))
    def forward(self, x):
        return self.spk_fc(x)
        
        
class SpeakerEncoder(BaseExtractor):
    def __init__(self, spk_ft_ckpt, spk_ft_size=1024, spk_emb_size=64,
                 speech_model='facebook/wav2vec2-large-xlsr-53', device='cuda', normalize=True,
                 sr=16000, ft_sr=50, source_extractor_config=None):
        
        if speech_model is not None:
            if 'wavlm' in speech_model:
                self.speech_model = WavLMModel.from_pretrained(speech_model)
            else:
                self.speech_model = Wav2Vec2Model.from_pretrained(speech_model)
            self.speech_model.encoder.layers = self.speech_model.encoder.layers[:1]
            self.speech_model = self.speech_model.eval().to(device)
        else:
            self.speech_model = None
        self.spk_enc = SpeakerEncodingLayer(spk_ft_size, spk_emb_size)
        ckpt = torch.load(spk_ft_ckpt, map_location="cpu")
        self.spk_enc.load_state_dict(ckpt)
        self.spk_enc = self.spk_enc.eval().to(device)
        self.device = device
        self.normalize= normalize
        self.sr = sr
        self.ft_sr = ft_sr
        if source_extractor_config is not None:
            self.source_extractor = SourceExtractor(**source_extractor_config)
        else:
            self.source_extractor = None
        
    def _extract_acoustics(self, wavs, outputs={}):
        if not isinstance(wavs, SpeechWave):
            wavs = self.process_wavfiles(wavs)
        with torch.no_grad():
            speech_outputs = self.speech_model(wavs.input_values,
                                        output_hidden_states=True)
        states=speech_outputs.hidden_states
        low_acoustics_ = states[0].cpu().numpy()
        outputs["wav"] = wavs
        outputs["acoustics"] = low_acoustics_
        return outputs
    
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
            periodicity = outputs["periodicity"][b][:ft_len]
            acoustics = outputs["acoustics"][b][:ft_len]
            spk_emb = self._get_spk_emb(acoustics, periodicity)
            spk_emb = torch.from_numpy(spk_emb).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                spk_emb = self.spk_enc(spk_emb)
            spk_emb = spk_emb.squeeze(0).cpu().numpy()
            split_outputs.append({"spk_emb": spk_emb})
        return split_outputs
    
    
    def __call__(self, wavfiles, outputs={}):
        wavs = self.process_wavfiles(wavfiles)
        if "periodicity" in outputs.keys():
            outputs = self.source_extractor._extract_pitch(wavs, outputs)
        outputs = self._extract_acoustics(wavs, outputs)
        outputs = self._split_batch(outputs)
        return outputs
    
    