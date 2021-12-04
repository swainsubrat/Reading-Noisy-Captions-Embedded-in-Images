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
from PIL import Image
from torchvision import transforms as T
from skimage import transform

from constants import model
import pandas as pd

from skimage.io import imread
from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pre_process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = transform.resize(image, (256, 256))
    image = T.ToTensor()(image)
    image = image.float().to(device)

    return image


def generate_captions(encoder, decoder, image_path, word_map, beam_size):

    vocab_size = len(word_map)
    image = pre_process_image(image_path)

    # Run Encoder
    # (1, 3, 256, 256)
    image = image.unsqueeze(0)
    # (1, 14, 14, 2048), where 2048 is the encoder_dim
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    # (1, 196, 2048)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)
    # (3, 196, 2048)
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)
    # First, the top k previous word is start
    # (3, 1)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(beam_size, 1).to(device)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)
    complete_seqs, complete_seqs_alpha, complete_seqs_scores = [], [], []

    # Run Decoder
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)


    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)

        prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode="floor") 
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            beam_size -= len(complete_inds)

        if beam_size == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

    top_k_scores = top_k_scores.tolist()
    idx = top_k_scores.index(max(top_k_scores))
    seq = seqs[idx]

    return seq


if __name__ == '__main__':

    imgs = "data/test_data/"
    model = f"data/BEST_{model}.pth.tar"
    beam_size = 3

    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    from utils import load

    _dict = load("./objects/processed_captions_training.pkl")
    word_map = _dict["word_map"]
    rev_word_map = {v: k for k, v in word_map.items()}
    
    submission = []
    dirs = os.listdir(imgs)
    dirs = sorted(dirs, key = lambda x: (len (x), x))

    for dir in dirs:
        img_path = f"./data/test_data/{dir}"
        seq = generate_captions(encoder, decoder, img_path, word_map, beam_size)
        seq = seq.tolist()[1:]
        length = np.random.randint(5, 7)
        seq = seq[:length]
        try:
            words = [rev_word_map[ind] for ind in seq]
            while "<unk>" in words:
                words.remove("<unk>")
            words = " ".join(words)
        except:
            words = ""
            print("Exception occured!!!")
        img_path = "test_data/" + dir
        submission.append([
            img_path, words
        ])
        print(words)
    
    df = pd.DataFrame(submission)
    df.to_csv("results.csv", sep="\t", index=False, header=None)
