import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizer.layers import LSQ_Conv2D, LSQ_Linear
from quantizer.lsq import LSQ_Quantizer


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


def QuantLeNet5(model, bits=8, n_classes=10):
    model.c1 = LSQ_Conv2D(model.c1, bits,
                          LSQ_Quantizer(8, False), LSQ_Quantizer(8, True))

    model.c3 = LSQ_Conv2D(model.c3,
                          bits,  LSQ_Quantizer(bits, False), LSQ_Quantizer(bits, True))

    model.c5 = LSQ_Conv2D(model.c5, bits,  LSQ_Quantizer(
        bits, False), LSQ_Quantizer(bits, True))

    model.f6 = LSQ_Linear(model.f6, bits,  LSQ_Quantizer(
        bits, False), LSQ_Quantizer(bits, True))

    model.f7 = LSQ_Linear(model.f7, 8,  LSQ_Quantizer(
        8, False), LSQ_Quantizer(8, True))

    return model
