import torch
from torch import nn

# Basic UNet implementation initially created by copilot and then modified
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        #Upsample layers
        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Decoder
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)
        
        # Output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        up4 = self.upsample4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        up3 = self.upsample3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        up2 = self.upsample2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        up1 = self.upsample1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))
        
        # Output
        output = self.output(dec1)
        
        return output