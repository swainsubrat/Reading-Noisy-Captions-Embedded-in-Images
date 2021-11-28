import yaml
from datetime import date

today = date.today()
with open("./configs/resnet50.yml") as file:
    config = yaml.load(file, yaml.SafeLoader)

# Model parameters
model         = config["model"]
emb_dim       = config["embedding_dim"]
attention_dim = config["attention_dim"]
decoder_dim   = config["decoder_dim"]
dropout       = config["dropout"]

# Training parameters
epochs    = config["epochs"]
workers   = config["workers"]
batch_size= config["batch_size"]
encoder_lr= config["encoder_lr"]
decoder_lr= config["decoder_lr"]
grad_clip = config["grad_clip"]
alpha_c   = config["alpha_c"]

# Dataset constants
caption_path    = "./data/Train_text.tsv"
image_path      = "./data/train_data"
test_image_path = "./data/test_data"

# File constants
log_file_path        = f"./logs/{model}.log"
checkpoint_path      = f"./data/{model}.pth.tar"
BEST_checkpoint_path = f"./data/BEST_{model}.pth.tar"

min_word_freq = config["min_word_freq"]
