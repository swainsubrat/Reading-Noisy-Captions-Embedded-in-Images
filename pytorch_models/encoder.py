import torch
from torch import nn
import torchvision

from constants import model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder for the task
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        if model == "resnet101":
            proposed_encoder = torchvision.models.resnet101(pretrained=True)
        elif model == "resnet50":
            proposed_encoder = torchvision.models.resnet50(pretrained=True)
        else:
            proposed_encoder = torchvision.models.vgg19(pretrained=True)

        modules = list(proposed_encoder.children())[:-2]
        self.proposed_encoder = nn.Sequential(*modules)

        # There may be images of different sizes, which will create problem while feeding to
        # the decoder. Hence, we're scaling them all to the same size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        out = self.adaptive_pool(
            self.proposed_encoder(images)
        )
        return out.permute(0, 2, 3, 1)

    def fine_tune(self, fine_tune=True):
        """
        set required grad to false first, then set it to true for some layers(as specified)
        """
        for p in self.proposed_encoder.parameters():
            p.requires_grad = False
        for c in list(self.proposed_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune