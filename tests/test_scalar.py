import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TinyTorch.primitives import Scalar

def test_scalar_add():
    result = Scalar(1.0) + Scalar(2.0)
    assert result.data == 3.0

def test_scalar_add_neg():
    result = Scalar(-1.0) + Scalar(-2.0)
    assert result.data == -3.0

def test_scalar_mul():
    result = Scalar(2.0) * Scalar(3.0)
    assert result.data == 6.0

def test_scalar_mul_neg():
    result = Scalar(-2.0) * Scalar(-3.0)
    assert result.data == 6.0

def test_scalar_pow():
    result = Scalar(2.0) ** 3
    assert result.data == 8.0

def test_scalar_pow_neg():
    result = Scalar(-2.0) ** -3
    assert result.data == -0.125

def test_scalar_sub():
    result = Scalar(2.0) - Scalar(1.0)
    assert result.data == 1.0

def test_scalar_sub_neg():
    result = Scalar(-2.0) - Scalar(-1.0)
    assert result.data == -1.0

def test_scalar_truediv():
    result = Scalar(4.0) / Scalar(2.0)
    assert result.data == 2.0

def test_scalar_truediv_neg():
    result = Scalar(-4.0) / Scalar(-2.0)
    assert result.data == 2.0

def test_scalar_autodiff():
    a = Scalar(2.0)
    b = Scalar(3.0)
    c = Scalar(4.0)
    y = a + (b * (c**2))
    y.backward()
    assert a.grad == 1.0 and b.grad == 16.0 and c.grad == 24.0

