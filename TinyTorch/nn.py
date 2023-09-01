import random
from TinyTorch.primitives import Tensor, matmul

class Linear:
    def __init__(self, inp, out, test=False):
        if test:
            random.seed(1337)
            self.weight = Tensor([[random.random() for _ in range(inp)] for _ in range(out)])
            self.bias = Tensor([[random.random()] for _ in range(out)])
        else:
            self.weight = Tensor([[random.random() for _ in range(inp)] for _ in range(out)])
            self.bias = Tensor([[0.0] for _ in range(out)])

    
    def __call__(self, x):
        output = matmul(self.weight, x)
        output += self.bias
        return output
    
    @property
    def parameters(self):
        return self.weight, self.bias
    
    @property
    def trainable(self):
        return True
    

class SequentialNN:
    def __init__(self, layers):
        self.layers = layers
        self.parameterlist = []

        for layer in self.layers:
            if layer.trainable:
                weight, bias = layer.parameters
                self.parameterlist.append(weight), self.parameterlist.append(bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze()
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    @property
    def parameters(self):
        return self.parameterlist