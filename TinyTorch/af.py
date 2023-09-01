from TinyTorch.primitives import Scalar
import math

class Tanh:
    def __call__(self, x):
        result = (Scalar(math.e) ** (Scalar(2) * x) - 1) / (Scalar(math.e) ** (Scalar(2) * x) + 1)

        def back():
            grad = 1 - result.data**2
            x.grad += grad * result.grad

        result.back = back

        return result
    
    @property
    def trainable(self):
        return False

class ReLU:
    def __call__(self, x):
        result = x * (x.data > 0)

        def back():
            grad = result.data > 0
            x.grad += grad * result.grad

        result.back = back

        return result
    
    @property
    def trainable(self):
        return False

class Sigmoid:
    def __call__(self, x):
        result = Scalar(1) / (Scalar(1) + Scalar(math.e) ** (x * -1))
    
        def back():
            grad = result.data * (1 - result.data)
            x.grad += grad * result.grad

        result.back = back
        
        return result
    
    @property
    def trainable(self):
        return False
    
class CustomAF:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative
    
    def __call__(self, x):
        result = self.function(x)

        def back():
            grad = self.derivative(x).data
            x.grad += grad * result.grad

        result.back = back

        return result
    
    @property
    def trainable(self):
        return False