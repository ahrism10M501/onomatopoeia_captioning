import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(int(out_channels/4), out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(int(out_channels/4), out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(int(out_channels/4), out_channels)
            )

    def forward(self, x):
        out = self.feature(x)
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out

class ObjectDetector(nn.Module):
    def __init__(self, num_classes=20):
        super(ObjectDetector, self).__init__()

        self.num_classes = num_classes
        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.res_block1 = ResidualBlock(in_channels=64, out_channels=128)
        self.res_block2 = ResidualBlock(in_channels=128, out_channels=128)

        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(8, 32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(32, 128),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(3)
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

        self.boxer = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.feature2(x)
        x= self.pool(x)
        x = x.view(x.size(0), -1)

        labels = self.fc(x)
        bbox = self.boxer(x)

        return labels, bbox