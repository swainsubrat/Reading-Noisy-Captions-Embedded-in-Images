import torch
from torch import nn

import numpy as np

from PIL import Image
from torchvision import transforms as T
from skimage import transform


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.encoded_image_size = encoded_image_size
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=2, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Not a VGG yb
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.ReLU(inplace = True)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
    def forward(self, images):
        out = self.model(images)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)

        return out

    def fine_tune(self, fine_tune=False):
        ...

if __name__ == '__main__':
    encoder = Encoder()
    device = "cpu"
    image_path = "data/tt/res27.jpg"
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = transform.resize(image, (256, 256))
    image = T.ToTensor()(image)
    image = image.float().to(device)

    # Encode
    image = image.unsqueeze(0) # (1, 3, 256, 256)
    encoder_out = encoder(image) # (1, enc_image_size, enc_image_size, encoder_dim)

    print(encoder_out.shape)