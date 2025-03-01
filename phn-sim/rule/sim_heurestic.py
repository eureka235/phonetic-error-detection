import json
import numpy as np
import pandas as pd

file_path = "phn_feature.json"

with open(file_path, "r", encoding="utf-8") as file:
    phoneme_features = json.load(file)

df = pd.DataFrame(phoneme_features)


weights = {
    "vc": 0.2,
    "vlng": 0.1,
    "vheight": 0.15,
    "vfront": 0.15,
    "vrnd": 0.1,
    "ctype": 0.2,
    "cplace": 0.2,
    "cvox": 0.1
}

total_weight = sum(weights.values())
normalized_weights = {key: value / total_weight for key, value in weights.items()}
weights = normalized_weights
print(normalized_weights)

n = len(df)
similarity_matrix = np.zeros((n, n))


for i in range(n):
    for j in range(n):
        if i == j:
            similarity_matrix[i, j] = 1 
        else:
            similarity = 0
            for feature, weight in weights.items():
                if df.loc[i, feature] == df.loc[j, feature]:
                    similarity += weight 
            similarity_matrix[i, j] = similarity

phoneme_similarity = pd.DataFrame(
    similarity_matrix, columns=df["phoneme"], index=df["phoneme"]
)


np.save("rule_sim_matrix.npy", similarity_matrix)

phoneme_similarity.to_csv("rule_sim_matrix.csv")