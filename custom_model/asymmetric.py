import torch.nn as nn
import torch


class AsymmetricNeck(nn.Module):
    def __init__(self, in_channels, out_channels, projection_ratio=4):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.depht_down = int(in_channels / projection_ratio)
        self.out_channels = out_channels
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.conv_layer1 = nn.Conv2d(in_channels = self.in_channels, out_channels = self.depht_down,
                               kernel_size = 1, stride = 1, padding = 0, bias = False)
        
        self.prelu1 = nn.PReLU()
        
        self.conv_layer21 = nn.Conv2d(in_channels = self.depht_down, out_channels = self.depht_down,
                                kernel_size = (1, 5), stride = 1, padding = (0, 2), bias = False)
        
        self.conv_layer22 = nn.Conv2d(in_channels = self.depht_down,
                                 out_channels = self.depht_down, kernel_size = (5, 1), stride = 1, padding = (2, 0), bias = False)
        
        self.prelu2 = nn.PReLU()
        
        self.conv_layer3 = nn.Conv2d(in_channels = self.depht_down,
                                out_channels = self.out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
        self.prelu3 = nn.PReLU()
        
        self.batchnorm = nn.BatchNorm2d(self.depht_down)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, tensor):

        main_tensor = tensor
        
        tensor = self.conv_layer1(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu1(tensor)
        
        tensor = self.conv_layer21(tensor)
        tensor = self.conv_layer22(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu2(tensor)
        
        tensor = self.conv_layer3(tensor)
                
        tensor = self.dropout(tensor)
        tensor = self.batchnorm2(tensor)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            
            extras = torch.zeros((tensor.size()[0], out_shape, tensor.shape[2], tensor.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            main_tensor = torch.cat((main_tensor, extras), dim = 1)
        
        tensor = tensor + main_tensor
        tensor = self.prelu3(tensor)
        
        return tensor
