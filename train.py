import sys
import time
import logging
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from torch import nn
from training import train, validate

from utils import *
from constants import *
from customDataset import *
from pytorch_models import Encoder
from decoder import Decoder


# Setting up logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler(log_file_path)
stdout_handler      = logging.StreamHandler(sys.stdout)
formatter           = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
output_file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
logger.addHandler(output_file_handler)
logger.addHandler(stdout_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
logger.info(f"Using {device} as the accelerator")

# Training parameters
start_epoch = 0
epochs_since_improvement = 0
best_bleu4 = 0.
best_bleu1 = 0.
print_freq = 100
fine_tune_encoder = True

try:
    torch.load(checkpoint_path)
except:
    checkpoint_path = None


def main():
    """
    Training and validation.
    """

    global best_bleu4, best_bleu1, epochs_since_improvement, checkpoint_path, start_epoch, fine_tune_encoder, model, word_map, rev_word_map

    _dict = load("./objects/processed_captions_training.pkl")
    word_map = _dict["word_map"]
    rev_word_map = {v: k for k, v in word_map.items()}

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        logger.info("Found Checkpoint :)")
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    else:
        logger.info("Couldn't Find Checkpoint :(")
        decoder = Decoder(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    ic_dataset = ImageAndCaptionsDataset()
    train_loader = torch.utils.data.DataLoader(
        ic_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=None)
    
    ic_dataset_val = ImageAndCaptionsDataset(
        caption_path="./objects/processed_captions_validation.pkl"
    )
    val_loader = torch.utils.data.DataLoader(
        ic_dataset_val, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=None)

    for epoch in range(start_epoch, epochs):

        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epoch % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        logger.info(f"Training for Epoch: {epoch+1} Done!!!!")

        recent_bleu4, recent_bleu1 = validate(val_loader=val_loader,
                                              encoder=encoder,
                                              decoder=decoder,
                                              criterion=criterion)

        is_best  = recent_bleu4 > best_bleu4
        is_best1 = recent_bleu1 > best_bleu1

        best_bleu4 = max(recent_bleu4, best_bleu4)
        best_bleu1 = max(recent_bleu1, best_bleu1)

        if not is_best1:
            epochs_since_improvement += 1
            logger.info("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(model, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


if __name__ == '__main__':
    main()