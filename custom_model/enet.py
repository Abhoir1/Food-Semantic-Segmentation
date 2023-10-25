from initiation import *
from downsample import *
from asymmetric import *
from upsample import *

class ENet(nn.Module):

    def __init__(self, C):

        super().__init__()
        self.C = C
        self.init = InitialBlock()
        
        self.bn10 = RDDNeck(dilation=1, in_channels=16, out_channels=64, down_flag=True, p=0.01)   
        
        self.bn11 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01)
        self.bn12 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01)
        self.bn13 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01)
        self.bn14 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, p=0.01)
        self.bn20 = RDDNeck(dilation=1, in_channels=64, out_channels=128, down_flag=True)
        self.bn21 = RDDNeck(dilation=1, in_channels=128, out_channels=128, down_flag=False)
        self.bn22 = RDDNeck(dilation=2, in_channels=128, out_channels=128, down_flag=False)
        
        self.bn23 = AsymmetricNeck(in_channels=128, out_channels=128)
        
        self.bn24 = RDDNeck(dilation=4, in_channels=128, out_channels=128, down_flag=False)
        self.bn25 = RDDNeck(dilation=1, in_channels=128, out_channels=128, down_flag=False)
        self.bn26 = RDDNeck(dilation=8, in_channels=128, out_channels=128, down_flag=False)
        
        self.bn27 = AsymmetricNeck(in_channels=128, out_channels=128)
        
        self.bn28 = RDDNeck(dilation=16, in_channels=128, out_channels=128, down_flag=False)
        self.bn31 = RDDNeck(dilation=1, in_channels=128, out_channels=128, down_flag=False)
        self.bn32 = RDDNeck(dilation=2, in_channels=128, out_channels=128, down_flag=False)
        
        self.bn33 = AsymmetricNeck(in_channels=128, out_channels=128)
        
        self.bn34 = RDDNeck(dilation=4, in_channels=128, out_channels=128, down_flag=False)
        self.bn35 = RDDNeck(dilation=1, in_channels=128, out_channels=128, down_flag=False)
        self.bn36 = RDDNeck(dilation=8, in_channels=128, out_channels=128, down_flag=False)
        
        self.bn37 = AsymmetricNeck(in_channels=128, out_channels=128)
        
        self.bn38 = RDDNeck(dilation=16, in_channels=128, out_channels=128, down_flag=False)
        
        self.bn40 = UBNeck(in_channels=128, out_channels=64, relu=True)
        
        self.bn41 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, relu=True)
        self.bn42 = RDDNeck(dilation=1, in_channels=64, out_channels=64, down_flag=False, relu=True)
        
        self.bn50 = UBNeck(in_channels=64, out_channels=16, relu=True)
        
        self.bn51 = RDDNeck(dilation=1, in_channels=16, out_channels=16, down_flag=False, relu=True)
        
        self.fullconv = nn.ConvTranspose2d(in_channels=16, out_channels=self.C, kernel_size=3, stride=2, padding=1, output_padding=1,bias=False)
        
        
    def forward(self, tensor):
        
        tensor = self.init(tensor)
        
        tensor, i1 = self.bn10(tensor)
        tensor = self.bn11(tensor)
        tensor = self.bn12(tensor)
        tensor = self.bn13(tensor)
        tensor = self.bn14(tensor)
        
        tensor, i2 = self.bn20(tensor)
        tensor = self.bn21(tensor)
        tensor = self.bn22(tensor)
        tensor = self.bn23(tensor)
        tensor = self.bn24(tensor)
        tensor = self.bn25(tensor)
        tensor = self.bn26(tensor)
        tensor = self.bn27(tensor)
        tensor = self.bn28(tensor)
        
        tensor = self.bn31(tensor)
        tensor = self.bn32(tensor)
        tensor = self.bn33(tensor)
        tensor = self.bn34(tensor)
        tensor = self.bn35(tensor)
        tensor = self.bn36(tensor)
        tensor = self.bn37(tensor)
        tensor = self.bn38(tensor)
        
        tensor = self.bn40(tensor, i2)
        tensor = self.bn41(tensor)
        tensor = self.bn42(tensor)
        
        tensor = self.bn50(tensor, i1)
        tensor = self.bn51(tensor)
        
        tensor = self.fullconv(tensor)
        
        return tensor
