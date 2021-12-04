import time
import logging
import torch.optim
import torch.utils.data

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from torch.nn.utils.rnn import pack_padded_sequence

from utils import *
from constants import *
from customDataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_freq = 100

_dict = load("./objects/processed_captions_training.pkl")
word_map = _dict["word_map"]
rev_word_map = {v: k for k, v in word_map.items()}

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    training
    """

    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores.data, targets.data)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(f"Epoch: {epoch}\t Loss: {losses.val}\t Avg Loss: {losses.avg}")


def validate(val_loader, encoder, decoder, criterion):
    """
    validation
    """
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = criterion(scores.data, targets.data)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print(f"Validation:\t Loss: {losses.val}\t Avg Loss: {losses.avg}")

            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps))
                references.append(img_captions)
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        ref_sentences = []
        for item in references:
            ref = item[0]
            words = [ rev_word_map[word] for word in ref ]
            sentence = list(" ".join(words))
            ref_sentences.append(sentence)

        hyp_sentenses = []
        for item in hypotheses:
            words = [ rev_word_map[word] for word in item ]
            sentence = list(" ".join(words))
            hyp_sentenses.append(sentence)
        
        score = np.mean([sentence_bleu([list(ref_sentences[i])], list(hyp_sentenses[i])) for i in range(len(hyp_sentenses))])

        print(f"Validation:\t Loss: {losses.val}\t Avg Loss: {losses.avg}\t BLEU-Character: {score}")

    return bleu4, score
