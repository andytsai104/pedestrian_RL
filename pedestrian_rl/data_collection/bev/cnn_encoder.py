import torch
import torch.nn as nn
'''
FEATURE STACKING (CNN Input):
   - The individual layers (Dict Keys) are stacked along the channel axis.
   - Example Output Shape: (320, 320, N) where N is the number of layers.
   - This "Multi-Channel Tensor" is fed directly into the CNN Encoder.

'''
class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels)
        )

    def forward(self, BEV_data):
        pass
