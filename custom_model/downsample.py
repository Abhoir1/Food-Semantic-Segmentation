import torch


class RDDNeck(torch.nn.Module):
    def __init__(self, dilation, in_channels, out_channels, down_flag, relu=False, projection_ratio=4, p=0.1):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        
        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag
        
        # calculating the number of reduced channels
        if down_flag:
            self.stride = 2
            self.depth_down = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.depth_down = int(out_channels // projection_ratio)
        
        if relu:
            activate_LU = torch.nn.ReLU()
        else:
            activate_LU = torch.nn.PReLU()
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices=True)
        
        self.dropout = torch.nn.Dropout2d(p=p)

        self.conv_layer = torch.nn.Conv2d(in_channels = self.in_channels, out_channels = self.depth_down, 
                               kernel_size = 1, stride = 1, padding = 0, bias = False, dilation = 1)
        
        self.prelu1 = activate_LU
        
        self.conv2 = torch.nn.Conv2d(in_channels = self.depth_down, out_channels = self.depth_down, 
                               kernel_size = 3, stride = self.stride, padding = self.dilation, bias = True, 
                               dilation = self.dilation)
                                  
        self.prelu2 = activate_LU
        
        self.conv3 = torch.nn.Conv2d(in_channels = self.depth_down, out_channels = self.out_channels, 
                               kernel_size = 1, stride = 1, padding = 0, bias = False, dilation = 1)
        
        self.prelu3 = activate_LU
        
        self.batchnorm = torch.nn.BatchNorm2d(self.depth_down)
        self.batchnorm2 = torch.nn.BatchNorm2d(self.out_channels)
        
        
    def forward(self, tensor):
        
        main_tensor = tensor
        
        tensor = self.conv_layer(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu1(tensor)
        
        tensor = self.conv2(tensor)
        tensor = self.batchnorm(tensor)
        tensor = self.prelu2(tensor)
        
        tensor = self.conv3(tensor)
        tensor = self.batchnorm2(tensor)
                
        tensor = self.dropout(tensor)
        
        if self.down_flag:
            main_tensor, id = self.maxpool(main_tensor)
          
        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            
            extras = torch.zeros((tensor.size()[0], out_shape, tensor.shape[2], tensor.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            main_tensor = torch.cat((main_tensor, extras), dim = 1)

        tensor = tensor + main_tensor
        tensor = self.prelu3(tensor)
        
        if self.down_flag:
            return tensor, id
        else:
            return tensor
