import torch
import torch.nn.functional as F


class LSQ_Conv2D(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, bits, weight_quantizer, act_quantizer):
        super(LSQ_Conv2D, self).__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=True if m.bias is not None else False,
            padding_mode=m.padding_mode
        )

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.bits = bits

        self.weight_quantizer = weight_quantizer
        self.weight_quantizer.init_step_size(m.weight)

        self.act_quantizer = act_quantizer

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)

        quantized_act = self.act_quantizer(x)

        # quantized_act = x

        return F.conv2d(quantized_act, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LSQ_Linear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Conv2d, bits, weight_quantizer, act_quantizer):
        super(LSQ_Linear, self).__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=True if m.bias is not None else False)

        self.weight = torch.nn.Parameter(m.weight.detach())
        self.bits = bits
        self.weight_quantizer = weight_quantizer

        self.weight_quantizer.init_step_size(m.weight)
        self.act_quantizer = act_quantizer

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_act = self.act_quantizer(x)

        # quantized_act = x

        return F.linear(quantized_act, quantized_weight, self.bias)
