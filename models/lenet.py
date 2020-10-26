import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        x = self.c1(x)
        x = F.relu(x)
        x = self.s2(x)

        x = self.c3(x)
        x = F.relu(x)
        x = self.s4(x)

        x = self.c5(x)
        x = F.relu(x)

        x = torch.squeeze(x)

        x = self.f6(x)
        x = F.relu(x)

        x = self.f7(x)

        return x
