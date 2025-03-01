import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import argparse
from model.inversion import Inversion


parser = argparse.ArgumentParser()
parser.add_argument("--rank",type=int,default=0)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--cuda",type=int,default=0)
parser.add_argument("--batch_size",type=int,default=1)

if __name__=='__main__':
    args = parser.parse_args()
    wav_dir =  Path('/data/cheoljun/FLEURS/FLEUR30_enhanced')
    save_dir = Path('/data/cheoljun/FLEURS/emasrc')
    save_dir.mkdir(exist_ok=True)
    spkft_save_dir = Path('/data/cheoljun/FLEURS/spk_ft')
    spkft_save_dir.mkdir(exist_ok=True)
    
    linear_model_path = '/home/cheoljun/GuBERT/linearmodels/wav2vec2-large-xlsr-53-17_cut-10_olm_linear.pkl'
    speech_model='facebook/wav2vec2-large-xlsr-53'
    target_layer=17
    freqcut=10
    pitch_q = 1
    device=f'cuda:{args.cuda}'
    inverter = Inversion(linear_model_path=linear_model_path,
                         speech_model=speech_model,
                         target_layer=target_layer,
                         freqcut=freqcut,
                         device=device,
                         pitch_q=pitch_q,
                        )
    
    batch_size = args.batch_size
    wav_files = []
    wav_files += [f for f in (wav_dir).glob('*.wav')]

    wav_files.sort()
    chunk_len = int(len(wav_files)/args.n)+1
    wav_files = wav_files[chunk_len*args.rank:chunk_len*(args.rank+1)]
    failed = []
    
    fidxs = [fi for fi in range(0,len(wav_files),batch_size)]
    for fi in tqdm.tqdm(fidxs):
        batch_wavfiles = []
        file_names = []
        spkft_file_names = []
        for wav_file in wav_files[fi:fi+batch_size]:
            file_name = save_dir/f'{wav_file.stem}.npy'
            spkft_file_name = spkft_save_dir/f'{wav_file.stem}.npy'
            if file_name.exists() and spkft_file_name.exists():
                continue
            file_names.append(file_name)
            batch_wavfiles.append(wav_file)
            spkft_file_names.append(spkft_file_name)
        
        if len(batch_wavfiles)==0:
            continue
        outputs = inverter(batch_wavfiles)
        for j in range(len(outputs)):
            np.save(file_names[j],outputs[j]["art"])
            np.save(spkft_file_names[j], outputs[j]["spk_emb"])
        