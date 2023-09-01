import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TinyTorch.primitives import Scalar, Tensor, matmul

def test_tensor_add():
    a = Tensor([[1.0,2.0],
               [3.0,4.0]])
    b = Tensor([[5.0,6.0],
               [7.0,8.0]])
    result = a + b
    assert result.data[0][0].data == 6.0 and result.data[0][1].data == 8.0 and result.data[1][0].data == 10.0 and result.data[1][1].data == 12.0

def test_matmul():
    a = Tensor([[1.0,2.0],
               [3.0,4.0]])
    b = Tensor([[5.0,6.0],
               [7.0,8.0]])
    result = matmul(a,b)
    assert result.data[0][0].data == 19.0 and result.data[0][1].data == 22.0 and result.data[1][0].data == 43.0 and result.data[1][1].data == 50.0

def test_tensor_shape():
    a = Tensor([[1.0,2.0],
               [3.0,4.0]])
    b = Tensor([[1.0,2.0,3.0],
               [3.0,4.0,5.0]])
    assert a.shape == (2,2) and b.shape == (2,3)