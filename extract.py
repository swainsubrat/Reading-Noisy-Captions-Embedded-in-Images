import os
import numpy as np
import pandas as pd

from utils import load


_dict = load("./objects/processed_captions_training.pkl")
word_map = _dict["word_map"]
rev_word_map = {v: k for k, v in word_map.items()}  # idx2word

idxs = list(rev_word_map.keys())[:-4]

submission = []

dirs = os.listdir("data/test_data/")
dirs = sorted(dirs, key = lambda x: (len (x), x))

for dir in dirs:

    length = np.random.randint(5, 7)

    caption = ""
    for len in range(length):
        idx = np.random.randint(1, 1978)
        word = rev_word_map[idx]
        if caption:
            caption = caption + " " + word
        else:
            caption = word
    
    submission.append([
        "test_data/" + dir, caption
    ])
from pprint import pprint
# pprint(submission)
df = pd.DataFrame(submission)
df.to_csv("submission.csv", sep="\t", index=False, header=None)
