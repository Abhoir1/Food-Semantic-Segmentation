import torch.nn as nn
import torch

class InitialBlock(nn.Module):
   
    def __init__ (self,in_channels = 3,out_channels = 13):
        
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 0)
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
        self.prelu = nn.PReLU(16)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, tensor):
        
        main = self.conv_layer(tensor)
        main = self.batchnorm(main)
        side = self.maxpool(tensor)
        
        tensor = torch.cat((main, side), dim=1)
        tensor = self.prelu(tensor)
        
        return tensor