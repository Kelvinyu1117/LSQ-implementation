import torch
import torch.tensor as Tensor
from brevitas.inject import BaseInjector


class LSQ_Quantizer(torch.nn.Module):
    def __init__(self, bit_width, is_activation=False):
        super(LSQ_Quantizer, self).__init__()

        self.bit_width = bit_width

        if(is_activation):
            self.Qn = 0
            self.Qp = 2 ** bit_width - 1
        else:
            self.Qn = -2**(bit_width - 1)
            self.Qp = 2 ** (bit_width - 1) - 1

        self.s = torch.nn.Parameter(torch.ones(1))

    def grad_scale(self, x, scale):
        y_out = x
        y_grad = x * scale

        y = torch.detach(y_out - y_grad) + y_grad

        return y

    def round_pass(self, x):
        y_out = x.round()
        y_grad = x
        y = torch.detach(y_out - y_grad) + y_grad

        return y

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor):

        scale_factor = 1 / (x.numel() * self.Qp) ** 0.5

        scale = self.grad_scale(self.s, scale_factor)
        x = x / scale
        x = x.clamp(self.Qn, self.Qp)

        x_bar = self.round_pass(x)

        x_hat = x_bar * scale

        return x_hat, self.s, torch.Tensor([self.bit_width])


class LSQ_weight_quant_8bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 8
    is_activation = False


class LSQ_input_quant_8bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 8
    is_activation = True


class LSQ_weight_quant_4bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 4
    is_activation = False


class LSQ_input_quant_4bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 4
    is_activation = True


class LSQ_weight_quant_2bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 2
    is_activation = False


class LSQ_input_quant_2bits(BaseInjector):
    tensor_quant = LSQ_Quantizer
    bit_width = 2
    is_activation = True
