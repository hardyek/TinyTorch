import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TinyTorch.nn import Linear, SequentialNN
from TinyTorch.primitives import Tensor, Scalar

def test_linear_forward():
    linear = Linear(2,3, test=True)
    x = Tensor([2.,3.])
    result = linear(x)
    print(result.data)
    assert result[0][0].data == 3.2190072433321526 and result[1][0].data == 3.278671605121253 and result[2][0].data == 3.7262223179967155

def test_sequential_forward():
    layer1 = Linear(1,3, test=True)
    layer2 = Linear(3,3, test=True)
    layer3 = Linear(3,1, test=True)
    layers = [layer1,layer2,layer3]

    NN = SequentialNN(layers)

    x = Scalar(2.0)

    result = NN(x)

    assert result.data.data == 5.330811307479268