import re
from collections import Counter
import pandas as pd
import json

def count_ipa_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()

    ipa_sequences = []
    for line in file_content.splitlines():
        parts = line.split('|')
        if len(parts) == 2:
            ipa_sequences.append(parts[1])

    all_phonemes = ''.join(ipa_sequences)
    phoneme_counts = Counter(all_phonemes)

    unique_ipa = set(''.join(phoneme_counts.keys()))
    ipa_set = f"ipa = {{{', '.join(repr(ipa) for ipa in sorted(unique_ipa))}}}"
    print(ipa_set)

    # ipa_data = {'ipa': sorted(unique_ipa)}
    # output_file = "ipa_phonemes.json"

    # with open(output_file, 'w', encoding='utf-8') as json_file:
    #     json.dump(ipa_data, json_file, ensure_ascii=False, indent=4)


    phoneme_df = pd.DataFrame(phoneme_counts.items(), columns=["IPA Phoneme", "Count"]).sort_values(by="Count", ascending=False)
    return phoneme_df


file_path = "/home/xuanru/tts/vits/filelists/ljs_audio_text_train_filelist.txt.cleaned" 
ipa_df = count_ipa_from_file(file_path)
print(ipa_df)

