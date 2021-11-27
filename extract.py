import os
import numpy as np
import pandas as pd

from utils import load


_dict = load("./objects/processed_captions.pkl")
word_map = _dict["word_map"]
rev_word_map = {v: k for k, v in word_map.items()}  # idx2word

idxs = list(rev_word_map.keys())[:-4]

submission = []

dirs = os.listdir("data/test_data/")
dirs = sorted(dirs, key = lambda x: (len (x), x))

for dir in dirs:

    length = np.random.randint(8, 10)

    caption = ""
    for len in range(length):
        idx = np.random.randint(1, 1990)
        word = rev_word_map[idx]
        if caption:
            caption = caption + " " + word
        else:
            caption = word
    
    submission.append([
        dir, caption
    ])

df = pd.DataFrame(submission)
# print(df)
df.to_csv("nc_submission.tsv", sep="\t")
