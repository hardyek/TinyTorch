import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TinyTorch.af import Tanh, ReLU, Sigmoid, CustomAF
from TinyTorch.primitives import Scalar, Tensor

def test_Tanh():
    x = Scalar(2)
    af = Tanh()
    result = af(x)
    result.backward()
    assert result.data == 0.9640275800758169 and x.grad == 0.07065082485316443

def test_ReLU():
    x = Scalar(2)
    af = ReLU()
    result = af(x)
    result.backward()
    assert result.data == 2. and x.grad == 1.

def test_Sigmoid():
    x = Scalar(2)
    af = Sigmoid()
    result = af(x)
    result.backward()
    assert result.data == 0.8807970779778823 and x.grad == 0.10499358540350662

def test_CustomAF():
    x = Scalar(3)

    def f(x):
        return x * x
    
    def fprime(x):
        return x * 2
    
    af = CustomAF(f,fprime)
    result = af(x)
    result.backward()
    assert result.data == 9. and x.grad == 6.

def test_elementwise():
    x = Tensor([2.,2.,2.])

    af = Tanh()

    result = x.elementwise(af)

    x = Tensor([[2.,2.,2.],[2.,2.,2.]])

    result2 = x.elementwise(af)

    assert result[0].data == 0.9640275800758169, result2[0][0].data == 0.9640275800758169 and result[1].data == result2[1][0].data
