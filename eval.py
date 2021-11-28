import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import pandas as pd

from PIL import Image
from torchvision import transforms as T
from skimage import transform

from skimage.io import imread
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    k = beam_size
    vocab_size = len(word_map)

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = transform.resize(image, (256, 256))
    image = T.ToTensor()(image)
    image = image.float().to(device)


    # Encode
    image = image.unsqueeze(0)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)

    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)

    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)

        awe, alpha = decoder.attention(encoder_out, h)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode="floor") 
        next_word_inds = top_k_words % vocab_size  # (s)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # "TODO": change remove below line
        complete_inds = incomplete_inds
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    # print(complete_seqs_scores)
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgs', '-i', help='path to images')
    parser.add_argument('--model', '-m', help='path to model')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    word_map = None
    try:
        with open(args.word_map, 'r') as j:
            word_map = json.load(j)
    except:
        from utils import load

        _dict = load("./objects/processed_captions_training.pkl")
        word_map = _dict["word_map"]
    rev_word_map = {v: k for k, v in word_map.items()}

    submission = []
    dirs = os.listdir(args.imgs)
    dirs = sorted(dirs, key = lambda x: (len (x), x))

    for dir in dirs:
        img_path = f"./data/test_data/{dir}"

        # Encode, decode with attention and beam search
        seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, 5)
        alphas = torch.FloatTensor(alphas)

        words = [rev_word_map[ind] for ind in seq]
        print(words)

        file_path = "test_data/" + dir

        submission.append([
            file_path, words
        ])
    df = pd.DataFrame(submission)
    df.to_csv("submission.csv", sep="\t", index=False, header=None)