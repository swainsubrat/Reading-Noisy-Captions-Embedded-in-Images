import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class ImageAndCaptionsDataset(Dataset):
    def __init__(self, img_path,caption_path, transform=None):
        self.image_filenames=sorted(os.listdir(img_path)) 
        self.captions=pd.read_csv(caption_path,sep='\t',header=None)
        print(self.image_filenames)
        # # print()
        # print(len(self.captions))

dataset=ImageAndCaptionsDataset("train_data","Train_text.tsv")