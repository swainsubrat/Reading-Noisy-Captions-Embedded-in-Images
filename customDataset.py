import os
import pickle
import numpy as np
import pandas as pd


import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

from skimage import transform

from utils import load

class ImageAndCaptionsDataset(Dataset):
    # caption_path is the processed caption path 
    def __init__(self, image_path="data/", caption_path="./objects/processed_captions_training.pkl", transform=None):
        super(ImageAndCaptionsDataset, self).__init__()
        self.transform = transform
        self.image_path=image_path
        _dict=load(caption_path)

        self.image_filenames=_dict["image_filenames"]
        self.caption_lengths=_dict["caption_lengths"]    
        self.max_length=_dict["max_caption_length"]
        self.captions=_dict["captions"]
        self.word_map=_dict["word_map"]

        
    def __getitem__(self, idx: int):

        image_path = os.path.join(self.image_path,self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = transform.resize(image, (256, 256))
        image = T.ToTensor()(image)
        image = image.float()

        caption = torch.LongTensor([self.captions[idx]])
        caption_length = torch.LongTensor([self.caption_lengths[idx]])
        self.cpi = 1
        all_captions = torch.LongTensor(
                self.captions[((idx // self.cpi) * self.cpi):(((idx // self.cpi) * self.cpi) + self.cpi)])
        return image, caption, caption_length, all_captions
 
    def __len__(self):
        return len(self.captions)


if __name__ == "__main__":
    ic_dataset = ImageAndCaptionsDataset()
    data_loader_test = torch.utils.data.DataLoader(
        ic_dataset, batch_size=5, shuffle=False, num_workers=4,
        collate_fn=None)

    for i, (imgs, caps, caplens) in enumerate(data_loader_test):
        print(imgs, caps, caplens)
        break
