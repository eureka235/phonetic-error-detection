import sys
sys.path.append('..')
from pathlib import Path
import numpy as np
import soundfile as sf
import tqdm
import argparse
from model.src_extractor import ProsodyExtractor


parser = argparse.ArgumentParser()
parser.add_argument("--rank",type=int,default=0)
parser.add_argument("--n",type=int,default=1)
parser.add_argument("--cuda",type=int,default=0)
parser.add_argument("--batch_size",type=int,default=1)


dirs = {"LibriTTS": ['/data/cheoljun/LibriTTS/wavs', '/data/cheoljun/LibriTTS/src'],
        "LibriTTS_R": ['/data/cheoljun/LibriTTS_R/wavs', '/data/cheoljun/LibriTTS_R/src'],
        "VCTK": ['/data/cheoljun/VCTK/wav', '/data/cheoljun/VCTK/src'],
        "FLEURS":['/data/cheoljun/FLEURS/FLEUR30_enhanced', '/data/cheoljun/FLEURS/src'],
        "EXPRESSO": ['/data/cheoljun/expresso/wavs_16k', '/data/cheoljun/expresso/src']}
if __name__=='__main__':
    args = parser.parse_args()
    
    for dataset in ["VCTK", "FLEURS", "EXPRESSO", "LibriTTS_R"]:
        wav_dir =  Path(dirs[dataset][0])

        spkft_save_dir = Path(dirs[dataset][1])
        spkft_save_dir.mkdir(exist_ok=True)

        device=f'cuda:{args.cuda}'
        inverter = ProsodyExtractor(device=device)

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
            spkft_file_names = []
            for wav_file in wav_files[fi:fi+batch_size]:
                spkft_file_name = spkft_save_dir/f'{wav_file.stem}.npy'
                if spkft_file_name.exists():
                    continue
                batch_wavfiles.append(wav_file)
                spkft_file_names.append(spkft_file_name)

            if len(batch_wavfiles)==0:
                continue
            outputs = inverter(batch_wavfiles)
            for j in range(len(outputs)):
                np.save(spkft_file_names[j], outputs[j])
