class Scalar:
    def __init__(self, data, dependencies=()):
        self.data = data
        self.dependencies = set(dependencies)
        self.grad = 0
        self.back = lambda: None
    
    #Utils
    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"
    
    @property
    def type(self):
        return "scalar"
    
    @property
    def shape(self):
        return (1,1)
    
    #Operations
    def __add__(self, addend):
        addend = addend if isinstance(addend, Scalar) else Scalar(addend)
        result = Scalar(self.data + addend.data, {self, addend})

        def back():
            self.grad += result.grad
            addend.grad += result.grad
        
        result.back = back

        return result
    
    def __mul__(self, factor):
        if isinstance(factor, list):
            factor = factor[0]
        factor_scalar = factor if isinstance(factor, Scalar) else Scalar(factor)
        result = Scalar(self.data * factor_scalar.data, {self, factor_scalar})

        def back():
            self.grad += factor_scalar.data * result.grad
            factor_scalar.grad += self.data * result.grad

        result.back = back

        return result
    
    #Pow does not update the exponents gradient.
    def __pow__(self, power):
        power = power if isinstance(power, Scalar) else Scalar(power)
        result = Scalar(self.data ** power.data, {self, power})

        def back():
            self.grad += (power.data * self.data**(power.data-1)) * result.grad

        result.back = back

        return result
    
    #Operations that are other operations.
    def __sub__(self, subtrahend):
        subtrahend_scalar = subtrahend if isinstance(subtrahend, Scalar) else Scalar(subtrahend)
        neg_subtrahend = Scalar(-1) * subtrahend_scalar
        result = self + neg_subtrahend

        result.dependencies.add(subtrahend_scalar)

        return result
    
    def __truediv__(self, other):
        other_scalar = other if isinstance(other, Scalar) else Scalar(other)
        result = self * other_scalar**-1

        def back():
            self.grad += result.grad / other_scalar.data
            other_scalar.grad -= result.grad * self.data / (other_scalar.data ** 2)

        result.back = back

        return result


    #Auto-diff
    def zero_grad(self):
        self.grad = 0
    
    def backward(self):
        graph = []
        visited = set()
        def build_graph(s):
            if s not in visited:
                visited.add(s)
                for node in s.dependencies:
                    build_graph(node)
                graph.append(s)
        build_graph(self)

        for s in graph:
            s.zero_grad()

        self.grad = 1
        for s in reversed(graph):
            s.back()

def array_to_scalars(input_array):
    if isinstance(input_array, Scalar):
        return input_array
    if isinstance(input_array, (int, float)):
        return Scalar(input_array)
    return [array_to_scalars(sub_array) for sub_array in input_array]

class Tensor:
    def __init__(self,data):
        self.data = array_to_scalars(data)

    def __getitem__(self, index):
        element = self.data[index]
        return element if isinstance(element, Scalar) else Tensor(element)
    
    def slice(self, index):
        return self.data[index]

    def __repr__(self):
        def scalar_to_value(scalar):
            if isinstance(scalar, Scalar):
                return scalar.data
            return [scalar_to_value(sub_scalar) for sub_scalar in scalar]

        formatted_data = scalar_to_value(self.data)
        formatted_data_str = "\n" + str(formatted_data).replace("],", "],\n")
        return f"Tensor(data={formatted_data_str})"
    
    @property
    def grads(self):
        def scalar_to_grad(scalar):
            if isinstance(scalar, Scalar):
                return scalar.grad
            return [scalar_to_grad(sub_scalar) for sub_scalar in scalar]

        formatted_data = scalar_to_grad(self.data)
        formatted_data_str = "\n" + str(formatted_data).replace("],", "],\n")
        print(f"Gradients(data={formatted_data_str})")

    @property
    def shape(self):
        def get_shape(tensor):
            return (
                [len(tensor)] + get_shape(tensor[0])
                if isinstance(tensor, (list, tuple))
                else []
            )

        return tuple(get_shape(self.data))
    
    @property
    def type(self):
        return "tensor"
    
    def __add__(self, addend):
        if isinstance(addend, Tensor):
            if self.shape != addend.shape:
                raise ValueError(f"Shapes incompatible for Tensor Addition. A:{self.shape}, B:{addend.shape}")

            result_data = [
                [element1 + element2 for element1, element2 in zip(row1, row2)]
                for row1, row2 in zip(self.data, addend.data)
            ]
        else:
            addend_scalar = addend if isinstance(addend, Scalar) else Scalar(addend)
            result_data = [[element + addend_scalar for element in row] for row in self.data]

        return Tensor(result_data)
    
    def elementwise(self, function):
        if isinstance(self.data[0],list):
            return Tensor([[function(element) for element in row] for row in self.data])
        else:
            return Tensor([function(element) for element in self.data])
        
    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self.shape) + dim + 1

        if dim < 0 or dim > len(self.shape):
            raise ValueError("Invalid dimension index")

        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        new_data = self.data
        for _ in range(dim, len(self.shape)):
            new_data = [new_data]

        return Tensor(new_data)
    
    def squeeze(self, dim=None):
        if dim is not None:
            if dim < 0:
                dim = len(self.shape) + dim 

            if dim < 0 or dim >= len(self.shape):
                raise ValueError("Invalid dimension index")

            if self.shape[dim] != 1:
                raise ValueError("Cannot squeeze a dimension with size other than 1")

            new_data = self.data

            for _ in range(len(self.shape) - 1, dim, -1):
                new_data = new_data[0]

        else:
            new_data = self.data

            while isinstance(new_data, list) and len(new_data) == 1:
                new_data = new_data[0]


        return Tensor(new_data)

    
def matmul(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shapes incompatible for Matrix Multiplication. A:{A.shape}, B:{B.shape}")

    columns = 1 if len(B.shape) == 1 else B.shape[1]

    result = []
    for row in A.data:
        row_result = []
        for column in range(columns):
            dot_product = Scalar(0)
            for i in range(len(row)):
                if columns > 1:
                    dot_product += row[i] * B.data[i][column]
                else:
                    dot_product += row[i] * B.data if isinstance(B, Scalar) else row[i] * B.data[i]
            row_result.append(dot_product)
        result.append(row_result)

    return Tensor(result)

def zeros(shape):
    array = []
    for _ in range(shape[0]):
        if len(shape) > 1:
            array.append(zeros(shape[1:]))
        else:
            array.append(0)

    return Tensor(array)