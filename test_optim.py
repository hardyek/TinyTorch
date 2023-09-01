import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TinyTorch.nn import Linear, SequentialNN
from TinyTorch.primitives import Tensor, Scalar
from TinyTorch.optim import SGD

def test_SGD():
    layer1 = Linear(1,3, test=True)
    layer2 = Linear(3,3, test=True)
    layer3 = Linear(3,1, test=True)
    layers = [layer1,layer2,layer3]

    NN = SequentialNN(layers)

    x = Scalar(2.0)

    result = NN(x)

    loss = (result.data - 5) ** 2

    optimiser = SGD(NN.parameters,0.01)

    loss.backward()

    optimiser.step()

    assert result.data.data == 5.330811307479268, layer1.weight.data[0].data == 0.6067120084423319