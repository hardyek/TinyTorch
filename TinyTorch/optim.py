from TinyTorch.primitives import Scalar, Tensor

class SGD():
    def __init__(self, parameters, learning_rate):
        self.parameterlist = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameterlist:
            if isinstance(parameter.data[0],list):
                parameter.data = [[Scalar((element.data - (element.grad * self.learning_rate))) for element in row] for row in parameter.data]
            else:
                parameter.data =  [Scalar((element.data - (element.grad * self.learning_rate))) for element in parameter.data]

