import torch
from lsq import LSQ_Quantizer


def replace_conv2D_module(model, new_conv2d, bits=8):
    for attr_str in dir(model):
        target = getattr(model, attr_str)
        if(type(target) == torch.nn.Conv2d):
            setattr(model, attr_str, new_conv2d(target, bits,
                                                LSQ_Quantizer(bits, False), LSQ_Quantizer(bits, True)))

    for name, module in model.named_children():
        replace_conv2D_module(module, new_conv2d)


def replace_linear_module(model, new_linear, bits=8):
    for attr_str in dir(model):
        target = getattr(model, attr_str)
        if(type(target) == torch.nn.Linear):
            setattr(model, attr_str, new_linear(target, bits,
                                                LSQ_Quantizer(bits, False), LSQ_Quantizer(bits, True)))

    for name, module in model.named_children():
        replace_linear_module(module, new_linear)
