import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2=nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.conv7=nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.conv8=nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True)
        )
        # Decoder (Deconvolutional Layers)
        
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.convT1=nn.Sequential(
            nn.ConvTranspose2d(4096, 2048, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.convT2=nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.convT3=nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.convT4=nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.convT5=nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.convT6=nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.convT7=nn.Sequential(
            nn.ConvTranspose2d(64, 8, 2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.convT8=nn.Sequential(
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=0),
            nn.Tanh()
        )

            
        

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass
        output = x
        for i in range(1,9):
            output = getattr(self, 'conv'+str(i))(output)
        for i in range(1,9):
            output = getattr(self, 'convT'+str(i))(output)
        # print(output.shape)
        # exit(0)
        
        return output
    