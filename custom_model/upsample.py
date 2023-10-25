import torch

class UBNeck(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.depth_down = int(in_channels / projection_ratio)
        self.out_channels = out_channels
        
        
        if relu:
            activate_LU = torch.nn.ReLU()
        else:
            activate_LU = torch.nn.PReLU()
        
        self.unpool = torch.nn.MaxUnpool2d(kernel_size = 2,
                                     stride = 2)
        
        self.main_conv_layer = torch.nn.Conv2d(in_channels = self.in_channels,
                                    out_channels = self.out_channels,
                                    kernel_size = 1)
        
        self.dropout = torch.nn.Dropout2d(p=0.1)
        
        
        self.conv_trans1 = torch.nn.ConvTranspose2d(in_channels = self.in_channels,
                               out_channels = self.depth_down,
                               kernel_size = 1,
                               padding = 0,
                               bias = False)
        
        
        self.prelu1 = activate_LU
        
        # This layer used for Upsampling
        self.conv_trans2 = torch.nn.ConvTranspose2d(in_channels = self.depth_down,
                                  out_channels = self.depth_down,
                                  kernel_size = 3,
                                  stride = 2,
                                  padding = 1,
                                  output_padding = 1,
                                  bias = False)
        
        self.prelu2 = activate_LU
        
        self.conv_trans3 = torch.nn.ConvTranspose2d(in_channels = self.depth_down,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  padding = 0,
                                  bias = False)
        
        self.prelu3 = activate_LU
        
        self.batchnorm = torch.nn.BatchNorm2d(self.depth_down)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.out_channels)
        
    def forward(self, tensor, indices):
        main_tensor = tensor
        
        # Side Branch
        tensor = self.conv_trans1(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu1(tensor)
        
        tensor = self.conv_trans2(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu2(tensor)
        
        tensor = self.conv_trans3(tensor)
        tensor = self.batchnorm2(tensor)
        
        tensor = self.dropout(tensor)
        
        # Main Branch
        
        main_tensor = self.main_conv_layer(main_tensor)
        main_tensor = self.unpool(main_tensor, indices, output_size=tensor.size())
        
        # summing the main and side branches
        tensor = tensor + main_tensor
        tensor = self.prelu3(tensor)
        
        return tensor
