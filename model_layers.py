import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.branch1 = ConvBlock(in_channels, n1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1),
            ConvBlock(n3x3_reduce, n3x3, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1),
            ConvBlock(n5x5_reduce, n5x5, padding=1),
            ConvBlock(n5x5, n5x5, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        return torch.cat([out1, out2, out3, out4], dim=1)

class LightweightGoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(LightweightGoogLeNet, self).__init__()

        # --- Stem Block (224x224 -> 28x28) ---
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7, stride=2, padding=3), # 112x112
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),               # 56x56
            ConvBlock(32, 32, kernel_size=1),
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                # 28x28
        )

        # --- Stage 3 (28x28) ---
        # 3a output: 32+64+16+16 = 128
        self.inception3a = InceptionModule(64, 32, 48, 64, 8, 16, 16)
        # 3b output: 64+96+16+32 = 208
        self.inception3b = InceptionModule(128, 64, 64, 96, 8, 16, 32)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # 14x14

        # --- Stage 4 (14x14) ---
        # 4a output: 96+104+24+32 = 256
        self.inception4a = InceptionModule(208, 96, 48, 104, 8, 24, 32)
        # 4b output: 80+112+32+32 = 256
        self.inception4b = InceptionModule(256, 80, 56, 112, 12, 32, 32)
        # 4c output: 64+128+32+32 = 256
        self.inception4c = InceptionModule(256, 64, 64, 128, 12, 32, 32)
        # 4d output: 56+144+32+32 = 264
        self.inception4d = InceptionModule(256, 56, 64, 144, 16, 32, 32)
        # 4e output: 128+160+48+64 = 400
        self.inception4e = InceptionModule(264, 128, 80, 160, 16, 48, 64)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # 7x7

        # --- Stage 5 (7x7) ---
        # 5a output: 128+160+48+64 = 400
        self.inception5a = InceptionModule(400, 128, 80, 160, 16, 48, 64)
        # 5b output: 192+192+48+48 = 480
        self.inception5b = InceptionModule(400, 192, 96, 192, 16, 48, 48)

        # --- Classifier ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(480, num_classes)


    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Stage 4
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Stage 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Output Head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x