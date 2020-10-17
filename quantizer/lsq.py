class LSQ_Quantizer(torch.nn.Module):
    def __init__(self, bits, is_activation=False):
        super(LSQ_Quantizer, self).__init__()

        self.bits = bits

        if(is_activation):
            self.Qn = 0
            self.Qp = 2 ** bits - 1
        else:
            self.Qn = -2**(bits - 1)
            self.Qp = 2 ** (bits - 1) - 1

        self.s = torch.nn.Parameter(torch.tensor(1.0))

    def init_step_size(self, x):
        self.s = torch.nn.Parameter(
            x.detach().abs().mean() * 2 / (self.Qp) ** 0.5)

    def grad_scale(self, x, scale):
        y_out = x
        y_grad = x * scale

        y = (y_out - y_grad).detach() + y_grad

        return y

    def round_pass(self, x):
        y_out = x.round()
        y_grad = x
        y = (y_out - y_grad).detach() + y_grad

        return y

    def forward(self, x):
        scale_factor = 1 / (x.numel() * self.Qp) ** 0.5

        s = self.grad_scale(self.s, scale_factor)
        x = x / s
        x = x.clamp(self.Qn, self.Qp)

        x_bar = self.round_pass(x)

        x_hat = x_bar * self.s

        return x_hat
