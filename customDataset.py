import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from torchvision import transforms as T
from PIL import Image

class ImageAndCaptionsDataset(Dataset):
    # caption_path is the processed caption path 
    def __init__(self, image_path="data/train_data",caption_path="processed_captions.tsv", transform=None):
        super(ImageAndCaptionsDataset, self).__init__()
        self.image_filenames
        self.image_path=image_path
        self.caption_path=caption_path
        dict=pickle.load(self.caption_path)

        self.image_filenames=dict["image_filenames"]
        self.caption_lengths=dict["caption_lengths"]    
        self.max_length=dict["max_caption_length"]
        self.captions=dict["captions"]
        self.word_map=dict["word_map"]

        
    def __getitem__(self, idx: int):
        image_path = os.path.join(self.image_path,self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")
        image = np.array(image) 
        image=T.ToTensor()(image)
        if self.transform is not None:
            image = self.transform(image)
        caption = torch.LongTensor(self.captions[idx])
        caption_length = torch.LongTensor([self.caption_lengths[idx]])
        return image,caption, caption_length
 
    def __len__(self):
        return len(self.captions)

