
from module import Module


class Alphas(Module):
    def __init__(self, alphas, dtda, name='alphas'):

        self.name = name
        self.alphas = alphas
        self.dtda = dtda

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return self.input_tensor

    def lrp(self, *args, **kwargs):
        if self.dtda:
            Rx = self.input_tensor * self.alphas
        else:
            Rx = self.alphas
        return Rx
