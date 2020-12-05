from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear
from brevitas.core.quant import QuantType
from quantizer.brevitas_implementation.lsq_brevitas import LSQ_weight_quant_8bits, LSQ_input_quant_8bits

weight_bit_width = 8
activation_width = 8


class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = QuantConv2d(
            1, 6, 5, weight_quant=None, input_quant=None, bias_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
        self.relu1 = QuantReLU(
            input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
        self.conv2 = QuantConv2d(6, 16, 5, weight_quant=None, input_quant=None, bias_quant=None,
                                 output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
        self.relu2 = QuantReLU(
            input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
        self.fc1 = QuantLinear(16*5*5, 120, bias=True,
                               weight_quant=None, input_quant=None, bias_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
        self.relu3 = QuantReLU(
            input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
        self.fc2 = QuantLinear(
            120, 84, bias=True, weight_quant=None, input_quant=None, bias_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
        self.relu4 = QuantReLU(
            input_quant=None, act_quant=None, output_quant=None, update_iqi=None, update_aqi=None)
        self.fc3 = QuantLinear(
            84, 10, bias=False, weight_quant=None, input_quant=None, bias_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out


def get_non_quantized_lenet():
    model = QuantLeNet()
    return model


def get_8_bits_quantized_lenet():
    model = QuantLeNet()
    model.conv1 = QuantConv2d(
        1, 6, 5, weight_quant=LSQ_weight_quant_8bits, bias_quant=None, input_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
    model.relu1 = QuantReLU(
        input_quant=None, act_quant=LSQ_input_quant_8bits, output_quant=None, update_iqi=None, update_aqi=None)

    model.conv2 = QuantConv2d(6, 16, 5, weight_quant=LSQ_weight_quant_8bits, bias_quant=None,
                              input_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)
    model.relu2 = QuantReLU(
        input_quant=None, act_quant=LSQ_input_quant_8bits, output_quant=None, update_iqi=None, update_aqi=None)

    model.fc1 = QuantLinear(16*5*5, 120, bias=True,
                            weight_quant=LSQ_weight_quant_8bits, bias_quant=None,
                            input_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)

    model.relu3 = QuantReLU(
        input_quant=None, act_quant=LSQ_input_quant_8bits, output_quant=None, update_iqi=None, update_aqi=None)

    model.fc2 = QuantLinear(
        120, 84, bias=True, weight_quant=LSQ_weight_quant_8bits, bias_quant=None,
        input_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)

    model.relu4 = QuantReLU(
        input_quant=None, act_quant=LSQ_input_quant_8bits, output_quant=None, update_iqi=None, update_aqi=None)

    model.fc3 = QuantLinear(
        84, 10, bias=False, weight_quant=LSQ_weight_quant_8bits, bias_quant=None,
        input_quant=None, output_quant=None, update_wqi=None, update_bqi=None, update_iqi=None, update_oqi=None)

    return model
