import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=)
        )

    def forward(self, BEV_data):
        pass
