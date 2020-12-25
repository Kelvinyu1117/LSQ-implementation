from torch.nn import Module
import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType


def make_qconv2d(in_planes, out_planes, kernel_size=1, padding=1, stride=1, groups=1, bias=True, no_quant=True, **kwargs) -> QuantConv2d:
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


def make_qrelu(no_quant=True, **kwargs) -> QuantReLU:
    if no_quant:
        return QuantReLU(input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
    else:
        return QuantReLU(bit_width=kwargs['bit_width'])


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, no_quant=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = make_qconv2d(
            in_planes=in_planes, out_planes=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = make_qconv2d(in_planes=out_planes, out_planes=out_planes, kernel_size=3,
                                  stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)

        self.act = make_qrelu(no_quant=no_quant)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                make_qconv2d(in_planes, self.expansion * out_planes,
                             kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )
