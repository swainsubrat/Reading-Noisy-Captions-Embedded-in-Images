import torch
from torch import nn
import torchvision
import os

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    
    def __init__(self, encoded_image_size=14):
        super(Encoder,self).__init__()
        self.enc_image_size=encoded_image_size
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        if not os.path.exists('saved_models/resnet101.pth'):
            resnet=torchvision.models.resnet101(pretrained=True)
            torch.save(resnet,'saved_models/resnet101.pth')
        else:
            print(f"Found model at saved_models/resnet101.pth, Loading it.....")
            resnet = torch.load('saved_models/resnet101.pth')
        
        print(list(resnet.children())[5:])
        #Remove linear and pool layer
        modules=list(resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        self.adaptive_pool=nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))
        self.fine_tune()

    def forward(self,images):
        out=self.resnet(images) # (batch_size, 2048, image_size/32, image_size/32)
        out=self.adaptive_pool(out) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out=out.permute(0,2,3,1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad=False
            
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad=fine_tune

encoder=Encoder(encoded_image_size=14)
# class Attention(nn.Module):
#     def __init__(self, encoder_dim,decoder_dim, attention_dim):
#         super(Attention,self).__init__()
#         self.