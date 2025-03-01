import re
from collections import Counter

input_file = "/home/xuanru/tts/vits/filelists/vctk_audio_sid_text_train_filelist.txt"  
output_file = "/home/xuanru/tts/vits/filelists/vctk_word.txt" 

word_counter = Counter()


with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        
        parts = line.strip().split('|')
        if len(parts) > 2:
            sentence = parts[2]
        
            words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence)
            word_counter.update(words)


with open(output_file, "w", encoding="utf-8") as file:
    for word in sorted(word_counter.keys()):
        file.write(f"{word}\n")

print(f"Finish, saved to {output_file}")
