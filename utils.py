import os
import pickle
import numpy as np
import pandas as pd

from constants import *
from pprint import pprint

from typing import List, Tuple
from collections import Counter


def get_word_map(captions: pd.Series) -> Tuple:
    word_freq = Counter()
    max_caption_length = 0
    caption_lengths=[]

    #TODO: Check for: hey, you -> "hey," "you"
    for cap in captions:

        words = cap.split(' ')
        words_length=len(words)
        caption_lengths.append(words_length)
        max_caption_length = max(words_length, max_caption_length)
        word_freq.update(words)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]

    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    return max_caption_length,caption_lengths, word_map


def pad_and_append(captions: pd.Series, max_caption_length: int) -> pd.Series:
    """
    To make all the token length equal with padding
    and append <start> and <end> in front and back
    """
    padded_captions = []
    for caption in captions:
        if len(caption.split()) <= max_caption_length:

            diff = max_caption_length - len(caption.split())
            to_append = " <pad>" * diff
            caption += to_append

        padded_captions.append("<start> " + caption + " <end>")
    
    return padded_captions

if __name__ == "__main__":
    df: pd.DataFrame    = pd.read_csv(caption_path ,names=["filenames", "captions"], sep='\t', header=None).head(50)
    captions: pd.Series = df["captions"]
    image_filenames=df["filenames"].to_list()
    max_caption_length,caption_lengths, word_map = get_word_map(captions)
    captions = pad_and_append(captions, max_caption_length)
    #dump the dict using pickle
    #int,list,dict,list,list
    dict={"max_caption_length":max_caption_length,"caption_lengths":caption_lengths,
    "word_map":word_map,"captions":captions,"image_filenames":image_filenames}

    print([len(c.split()) for c in captions])
    print(captions)
