from torch import nn



import torch


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture matching the experiments of https://arxiv.org/abs/2205.13648.
    """

    def __init__(self, in_size=(32, 32, 3), out_size=10):
        super(SimpleCNN, self).__init__()

        assert len(in_size) == 3
        w, h, c = in_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        feature_size = ((((w+1)//2)+1)//2) * ((((h+1)//2)+1)//2) * 64
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_size)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        flat = conv_out.view(conv_out.shape[0], -1)
        fc_out = self.fc(flat)
        return fc_out




class CIFAR_CNN(nn.Module):
    def __init__(self, num_layers=3, num_classes=10):
        super(CIFAR_CNN, self).__init__()
        assert 3 <= num_layers <= 7, "num_layers must be between 3 and 7"

        layers = []
        in_channels = 3
        out_channels = 32
        pool_count = 0

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU())
            if i % 2 == 1:  # Add MaxPool every 2 layers
                layers.append(nn.MaxPool2d(2))
                pool_count += 1
            in_channels = out_channels  # keep it 32

        self.features = nn.Sequential(*layers)

        # Image starts as 32x32. Each MaxPool2d(2) halves it.
        spatial_dim = 32 // (2 ** pool_count)
        flat_dim = out_channels * spatial_dim * spatial_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  