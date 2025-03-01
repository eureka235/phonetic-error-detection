import torch
import random
import re
import json
import csv
import math

from text import symbols
from enum import Enum
from generate_utils import *
from models import SynthesizerTrn
import utils
import pysle
import gc
import sys
from tqdm import tqdm
from text import text_to_sequence

import setproctitle
setproctitle.setproctitle("simulation")

device =  torch.device("cuda:4")

def get_text_word(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_text_phn(text, hps):
    text_norm = use_phoneme(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def g2p(text):
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    return phonemes



def ipa_to_cmu_conversion(ipa_str):
    ipa_to_cmu = {"a": "AA", "b": "B", "d": "D", "e": "EY", "f": "F", "h": "HH",
                  "i": "IY", "j": "Y", "k": "K", "l": "L", "m": "M", "n": "N",
                  "o": "OW", "p": "P", "r": "R", "s": "S", "t": "T", "u": "UW",
                  "v": "V", "w": "W", "z": "Z", "æ": "AE", "ð": "DH", "ŋ": "NG",
                  "ɐ": "AH", "ɑ": "AA", "ɔ": "AO", "ə": "AH", "ɚ": "ER", "ɛ": "EH",
                  "ɜ": "ER", "ɡ": "G", "ɪ": "IH", "ɫ": "L", "ɹ": "R", "ɾ": "D", 
                  "ʃ": "SH", "ʊ": "UH", "ʌ": "AH", "ʒ": "ZH", "θ": "TH", "ᵻ": "IH"
    }

    vowel_mapping = {"a": "AA", "e": "EY", "i": "IY", "o": "OW", "u": "UW", 
                     "æ": "AE", "ɐ": "AH", "ɑ": "AA", "ɔ": "AO", "ə": "AH", 
                     "ɚ": "ER", "ɛ": "EH", "ɜ": "ER", "ɪ": "IH", "ʊ": "UH", 
                     "ʌ": "AH", "ᵻ": "IH"}
    
    consonant_mapping = {"b": "B", "d": "D", "f": "F", "h": "HH", "j": "Y", "k": "K", 
                        "l": "L", "m": "M", "n": "N", "p": "P", "r": "R", "s": "S", 
                        "t": "T", "v": "V", "w": "W", "z": "Z", "ð": "DH", "ŋ": "NG", 
                        "ɡ": "G", "ɫ": "L", "ɹ": "R", "ɾ": "DX", "ʃ": "SH", "ʒ": "ZH", 
                        "θ": "TH"}

    result = []
    i = 0
    while i < len(ipa_str):
        # double
        if ipa_str[i:i+2] in ipa_to_cmu:
            result.append(ipa_to_cmu[ipa_str[i:i+2]])
            # print(f"{ipa_str[i:i+2]} -> {ipa_to_cmu[ipa_str[i:i+2]]}")
            i += 2
        elif ipa_str[i] in ipa_to_cmu:
            result.append(ipa_to_cmu[ipa_str[i]])
            # print(f"{ipa_str[i]} -> {ipa_to_cmu[ipa_str[i]]}")
            i += 1
        else:
            # print(f"{ipa_str[i]} -> sil")
            i += 1
    
    return ' '.join(result)


CMU_dict = ["SIL", "OW", "UW", "EY", "AW", "AH", "AO", "AY", "EH", 
            "K", "NG", "F", "JH", "M", "CH", "IH", "UH", "HH",
            "L", "AA", "R", "TH", "AE", "D", "Z", "OY", "DH", 
            "IY", "B", "W", "S", "T", "SH", "ZH", "ER", "V", 
            "Y", "N", "G", "P"] # 40

ipa_dict = ["a", "b", "d", "e", "f", "h", "i", "j", "k", "l", "m", "n", "o", 
            "p", "r", "s", "t", "u", "v", "w", "z", "æ", "ð", "ŋ", "ɐ", "ɑ", 
            "ɔ", "ə", "ɚ", "ɛ", "ɜ", "ɡ", "ɪ", "ɫ", "ɹ", "ɾ", "ʃ", "ʊ", "ʌ", 
            "ʒ", "θ", "ᵻ"] # 42

vowel_replacements = {
    "a": ["i"], "i": ["a"],
    "æ": ["u"], "u": ["æ"],
    "ɑ": ["ɪ"], "ɪ": ["ɑ"],
    "o": ["ɛ"], "ɛ": ["o"],
    "ɔ": ["e"], "e": ["ɔ"],
    "ʊ": ["ɜ"], "ɜ": ["ʊ"],
    "ə": ["i"], "i": ["ə"],
    "ɚ": ["o"], "o": ["ɚ"],
    "ʌ": ["æ"], "æ": ["ʌ"]
}

consonant_replacements = {
    "p": ["ɡ"], "ɡ": ["p"],
    "t": ["ʒ"], "ʒ": ["t"],
    "k": ["b"], "b": ["k"],
    "m": ["s"], "s": ["m"],
    "n": ["ʃ"], "ʃ": ["n"],
    "ŋ": ["f"], "f": ["ŋ"],
    "l": ["t"], "t": ["l"],
    "ɹ": ["d"], "d": ["ɹ"],
    "w": ["k"], "k": ["w"],
    "θ": ["v"], "v": ["θ"],
    "ð": ["z"], "z": ["ð"],
    "ʃ": ["h"], "h": ["ʃ"]
}

def sub_pair_ipa(ipa_sequence, choice):
    ipa_list = list(ipa_sequence)
    
    vowel_positions = [i for i, char in enumerate(ipa_list) if char in vowel_replacements]
    consonant_positions = [i for i, char in enumerate(ipa_list) if char in consonant_replacements]
    
    if not (vowel_positions or consonant_positions):
        return ipa_sequence
    
    
    if choice == 'vowel':
        position = random.choice(vowel_positions)
        current_vowel = ipa_list[position]
        ipa_list[position] = random.choice(vowel_replacements[current_vowel])
    else:
        position = random.choice(consonant_positions)
        current_consonant = ipa_list[position]
        ipa_list[position] = random.choice(consonant_replacements[current_consonant])
    
    return ''.join(ipa_list)



# def generate_time_label(durations, stn_tst, cmu_seq):
#     vits_unit = 256 / 22050
#     target_unit = 0.02

#     result = torch.cat([torch.tensor([val] * int(rep)) for val, rep in zip(stn_tst, durations)])

#     skip_values = [16, 158, 157, 156] # [blank, ',:]
#     result = torch.where(torch.isin(result, torch.tensor(skip_values)), 0, result)
#     # print(result)

#     cmu_seq_str = cmu_seq.split()
#     # print(cmu_seq_str)

#     seen = set()
#     unique_values = torch.tensor([v.item() for v in result if v != 0 and (v.item() not in seen and not seen.add(v.item()))])
#     # print(unique_values)
    
#     length = len(cmu_seq_str)
#     # phoneme_to_index = {phoneme: CMU_dict.index(phoneme) for phoneme in cmu_seq_str}
    
#     phoneme_to_index = [
#         (cmu_seq_str[i], CMU_dict.index(cmu_seq_str[i])) for i in range(length)
#         ]

#     mapping = {i: phoneme_to_index[i][1] for i in range(len(phoneme_to_index))}

#     result_mapped = torch.tensor([
#         mapping[i] if i in mapping else 0 for i in range(len(phoneme_to_index))
#     ])

#     original_label = result_mapped
#     scaling_factor = vits_unit / target_unit
#     new_length = math.ceil(len(original_label) * scaling_factor)

#     new_label = torch.zeros(new_length, dtype=torch.int32)

#     for i in range(new_length):
#         original_index = int(i / scaling_factor)
#         if original_index < len(original_label):
#             new_label[i] = original_label[original_index]    

#     return new_label


def generate_time_label(durations, stn_tst, cmu_seq):
    vits_unit = 256 / 22050
    target_unit = 0.02
    scaling_factor = vits_unit / target_unit

    result = torch.cat([torch.tensor([val] * int(rep)) for val, rep in zip(stn_tst, durations)])
    ipa_label = result
    # print(ipa_label)

    skip_values = [16, 158, 157, 156] # [blank, ',:]
    result = torch.where(torch.isin(result, torch.tensor(skip_values)), 0, result)

    cmu_seq_str = cmu_seq.split()

    seen = set()
    unique_values = torch.tensor([v.item() for v in result if v != 0 and (v.item() not in seen and not seen.add(v.item()))])

    phoneme_to_index = {phoneme: CMU_dict.index(phoneme) for phoneme in cmu_seq_str}

    mapping = {unique_values[i].item(): list(phoneme_to_index.values())[i] for i in range(len(unique_values))}
    result_mapped = torch.tensor([mapping[val.item()] if val.item() in mapping else 0 for val in result])

    cmu_label = result_mapped
    # print(cmu_label)
    new_length = math.ceil(len(cmu_label) * scaling_factor)

    cmu_label_scal = torch.zeros(new_length, dtype=torch.int32)
    ipa_label_scal = torch.zeros(new_length, dtype=torch.int32)

    
    for i in range(new_length):
        original_index = int(i / scaling_factor)
        next_index = int((i + 1) / scaling_factor)
        
        if original_index < len(cmu_label):
            cmu_label_scal[i] = cmu_label[original_index]
            ipa_label_scal[i] = ipa_label[original_index]
        
        if next_index < len(cmu_label):
            segment_cmu = cmu_label[original_index:next_index + 1]
            segment_ipa = ipa_label[original_index:next_index + 1]
            if torch.any(segment_cmu > 0):  
                cmu_label_scal[i] = segment_cmu[segment_cmu > 0][0]
                ipa_label_scal[i] = segment_ipa[segment_ipa > 0][0]


    return cmu_label_scal, ipa_label_scal



def generate_sub_word_vowel(text, net_g, hps, out_file, sid):
    gt_ipa_seq = g2p(text)
    gt_cmu_seq = ipa_to_cmu_conversion(gt_ipa_seq)

    # sub vowel
    sub_ipa_seq = sub_pair_ipa(gt_ipa_seq, "vowel")
    sub_cmu_seq = ipa_to_cmu_conversion(sub_ipa_seq)
    
    # gen audio
    stn_tst = get_text_phn(sub_ipa_seq, hps)
    
    audio, durations = infer_audio_new(stn_tst, net_g, sid, device, len_scale=1.3)

    cmu_label, ipa_label = generate_time_label(durations, stn_tst, sub_cmu_seq)
    
    cmu_path = out_file.replace("audio", "cmu")
    cmu_path = cmu_path.replace(".wav", ".npy")

    ipa_path = out_file.replace("audio", "ipa")
    ipa_path = ipa_path.replace(".wav", ".npy")
    
    np.save(cmu_path, cmu_label.cpu().numpy())
    np.save(ipa_path, ipa_label.cpu().numpy())


    label = [{
        "error": True,
        "sub_cmu_seq": sub_cmu_seq,
        "sub_ipa_seq": sub_ipa_seq,
        "gt_word": text,
        "gt_cmu_seq": gt_cmu_seq,
        "gt_ipa_seq": gt_ipa_seq
    }]
    
    json_path = out_file.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)


def generate_sub_word_consonant(text, net_g, hps, out_file, sid):
    gt_ipa_seq = g2p(text)
    gt_cmu_seq = ipa_to_cmu_conversion(gt_ipa_seq)

    # sub vowel
    sub_ipa_seq = sub_pair_ipa(gt_ipa_seq, "con")
    sub_cmu_seq = ipa_to_cmu_conversion(sub_ipa_seq)
    
    # gen audio
    stn_tst = get_text_phn(sub_ipa_seq, hps)
    
    audio, durations = infer_audio_new(stn_tst, net_g, sid, device, len_scale=1.3)
    
    cmu_label, ipa_label = generate_time_label(durations, stn_tst, sub_cmu_seq)
    
    cmu_path = out_file.replace("audio", "cmu")
    cmu_path = cmu_path.replace(".wav", ".npy")

    ipa_path = out_file.replace("audio", "ipa")
    ipa_path = ipa_path.replace(".wav", ".npy")
    
    np.save(cmu_path, cmu_label.cpu().numpy())
    np.save(ipa_path, ipa_label.cpu().numpy())


    label = [{
        "error": True,
        "sub_cmu_seq": sub_cmu_seq,
        "sub_ipa_seq": sub_ipa_seq,
        "gt_word": text,
        "gt_cmu_seq": gt_cmu_seq,
        "gt_ipa_seq": gt_ipa_seq
    }]
    
    json_path = out_file.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)



def generate_word(text, net_g, hps, out_file, sid):
    gt_ipa_seq = g2p(text)
    gt_cmu_seq = ipa_to_cmu_conversion(gt_ipa_seq)

    # sub vowel
    sub_ipa_seq = gt_ipa_seq
    sub_cmu_seq = gt_cmu_seq
    
    # gen audio
    stn_tst = get_text_phn(sub_ipa_seq, hps)
    
    audio, durations = infer_audio_new(stn_tst, net_g, sid, device, len_scale=1.3)

    cmu_label, ipa_label = generate_time_label(durations, stn_tst, sub_cmu_seq)
    
    cmu_path = out_file.replace("audio", "cmu")
    cmu_path = cmu_path.replace(".wav", ".npy")

    ipa_path = out_file.replace("audio", "ipa")
    ipa_path = ipa_path.replace(".wav", ".npy")
    
    np.save(cmu_path, cmu_label.cpu().numpy())
    np.save(ipa_path, ipa_label.cpu().numpy())


    label = [{
        "error": False,
        "sub_cmu_seq": sub_cmu_seq,
        "sub_ipa_seq": sub_ipa_seq,
        "gt_word": text,
        "gt_cmu_seq": gt_cmu_seq,
        "gt_ipa_seq": gt_ipa_seq
    }]
    
    json_path = out_file.replace("audio", "labels")
    with open(json_path.replace(".wav", ".json"), 'w') as f:
        json.dump(label, f, indent=4)

    write_audio_from_np(audio, out_file)



def generate(text, net_g, hps, out_path, sid):
    # generate_sub_word_vowel(text, net_g, hps, out_path.replace(".wav", "_v.wav"), sid)
    # generate_sub_word_consonant(text, net_g, hps, out_path.replace(".wav", "_c.wav"), sid)
    generate_word(text, net_g, hps, out_path, sid)


if __name__ == "__main__": 
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint("./path/to/pretrained_vctk.pth", net_g, None)


    # word dict
    word_list = "filelists/vctk_word.txt"
    for speaker_id in tqdm(range(56, 109), desc="processing speakers"): # (1, 56) (56, 109)
        sid = torch.LongTensor([speaker_id]).to(device) 

        with open(word_list, mode='r') as file:
            lines = file.readlines() 
            for file_idx, line in enumerate(
                tqdm(lines, desc=f"Speaker {speaker_id}", leave=False), start=1
                ):
                
                word = line.strip()
                if not word:
                    continue
                
                filename = f"{speaker_id:03}_{file_idx:04}.wav"
    
                out_path = f"/data/xuanru/VCTK_accent/audio/{filename}"
                # generate(word, net_g, hps, out_path=out_path, sid=sid)

                try:
                    generate(word, net_g, hps, out_path=out_path, sid=sid)
                except Exception:
                    print("error:{}".format(out_path))
                    continue
            
        torch.cuda.empty_cache()
        gc.collect()

