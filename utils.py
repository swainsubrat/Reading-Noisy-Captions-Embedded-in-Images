import os
import torch
import pickle
import numpy as np
import pandas as pd

from constants import *
from pprint import pprint

from typing import Any, List, Tuple
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

def encoded_captions(captions,caption_lengths,word_map, max_caption_length):
    enc_captions = []
    for i,caption in enumerate(captions):
        enc_caption = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in caption.split()] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_caption_length - caption_lengths[i])

        enc_captions.append(enc_caption)
    return enc_captions

def save(obj: Any, path: str) -> None:
    outfile = open(path, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()

def load(path: str) -> None:
    infile = open(path, 'rb')
    obj = pickle.load(infile)
    infile.close()

    return obj

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

if __name__ == "__main__":
    df: pd.DataFrame    = pd.read_csv(caption_path ,names=["filenames", "captions"], sep='\t', header=None).head(45000)
    captions: pd.Series = df["captions"]
    image_filenames     = df["filenames"].to_list()

    max_caption_length, caption_lengths, word_map = get_word_map(captions)
    # padded_captions = pad_and_append(captions, max_caption_length)
    encoded_caption = encoded_captions(captions, caption_lengths, word_map, max_caption_length)

    _dict = {
        "max_caption_length": max_caption_length,
        "caption_lengths": caption_lengths,
        "word_map": word_map,
        "captions": encoded_caption,
        "image_filenames": image_filenames
    }

    save(_dict, "./objects/processed_captions.pkl")
