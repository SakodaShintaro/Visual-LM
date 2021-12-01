import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet101",
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
