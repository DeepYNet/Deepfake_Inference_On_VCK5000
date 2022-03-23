""" Full assembly of the parts to form the complete network """

from .unet_parts import *


# +
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.down5 = Down(1024, 2048)
        
#         self.linear_1 = nn.Linear(2048, 1024)
        self.linear_1 = nn.Linear(1024, 512)
        self.linear_2 = nn.Linear(512, 1)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        

        
        classification_part = F.max_pool2d(x5, kernel_size=[2,2])
        classification_part = F.max_pool2d(classification_part, kernel_size=[7,7])
        classification_part = torch.flatten(classification_part, start_dim=1)
        classification_part = F.relu(self.linear_1(classification_part))

        classification_result = torch.sigmoid(self.linear_2(classification_part))
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, classification_result
