from torch.nn import Module
import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.size())  # print(x.shape)
        return x


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or issubclass(type(m), nn.Linear) or issubclass(type(m), nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def make_qconv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=1, groups=1, bias=True, no_quant=True, **kwargs) -> QuantConv2d:
    if no_quant:
        return QuantConv2d(in_channels=in_planes,
                           out_channels=out_planes,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias,
                           weight_quant=None,
                           input_quant=None,
                           bias_quant=None,
                           output_quant=None,
                           update_wqi=None,
                           update_bqi=None,
                           update_iqi=None,
                           update_oqi=None)
    else:
        return QuantConv2d(in_channels=in_planes,
                           out_channels=out_planes,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           groups=groups,
                           bias=bias,
                           weight_bit_width=kwargs['bit_width'])


def make_qlinear(in_features, out_features, bias=True, no_quant=True, **kwargs) -> QuantConv2d:
    if no_quant:
        return QuantLinear(in_features=in_features,
                           out_features=out_features,
                           bias=bias,
                           weight_quant=None,
                           input_quant=None,
                           bias_quant=None,
                           output_quant=None,
                           update_wqi=None,
                           update_bqi=None,
                           update_iqi=None,
                           update_oqi=None)
    else:
        return QuantLinear(in_features=in_features,
                           out_features=out_features,
                           bias=bias,
                           weight_bit_width=kwargs['bit_width'])


def make_qrelu(no_quant=True, **kwargs) -> QuantReLU:
    if no_quant:
        return QuantReLU(input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
    else:
        return QuantReLU(bit_width=kwargs['bit_width'])


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, no_quant=True, bit_width=8):
        super(BasicBlock, self).__init__()
        self.conv1 = make_qconv2d(
            in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=stride, padding=1, bias=False, bit_width=bit_width)

        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = make_qconv2d(in_planes=out_planes, out_planes=out_planes, kernel_size=3,
                                  stride=1, padding=1, bias=False, bit_width=bit_width)

        self.bn2 = nn.BatchNorm2d(out_planes)

        self.act = make_qrelu(no_quant=no_quant, bit_width=bit_width)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            print((in_planes, self.expansion * out_planes, stride))
            self.shortcut = nn.Sequential(
                make_qconv2d(in_planes, self.expansion * out_planes,
                             kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(identity)

        out += shortcut

        out = self.act(out)

        return out


class QuantResNet(Module):
    def __init__(self, block: BasicBlock, num_blocks, num_classes=10, no_quant=True, bit_width=8):
        super(QuantResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = make_qconv2d(3, 16, kernel_size=3,
                                  stride=1, padding=1, bias=False, no_quant=no_quant, bit_width=bit_width)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, no_quant=no_quant, bit_width=bit_width)

        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, no_quant=no_quant, bit_width=bit_width)

        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, no_quant=no_quant, bit_width=bit_width)

        self.linear = make_qlinear(
            64, num_classes, no_quant=no_quant, bit_width=bit_width)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, no_quant=True, bit_width=8):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes=self.in_planes, out_planes=planes, stride=stride,
                                no_quant=no_quant, bit_width=bit_width))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = nn.functional.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(no_quant=True, bit_width=8):
    return QuantResNet(BasicBlock, [3, 3, 3], num_classes=10, no_quant=no_quant, bit_width=bit_width)
