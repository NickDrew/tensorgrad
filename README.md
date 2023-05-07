
# tensorgrad

![awww](puppies.jpg)

A derivative of micrograd, altered to process multidimensional arrays of scalar values.


### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value([[-4.0,-8.0],[-2.0,-4.0]])
b = Value([[2.0,6.0],[4.0,8.0]])
c = a + b
d = a * b + b**3 #applies the pow 3 to all values in the tensor
c += c + 1 #auto-converts the scalar 1 into a gradient-tracked tensor value of [[1,1],[1,1]] to match the shape of the tensor in c
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0/ f
print(g.data) # prints the outcome of the forward pass for all four values in the tensor
g.backward()
print(a.grad) # prints the numerical value of dg/d for all four values in the tensor
print(b.grad) # prints the numerical value of dg/db for all four values in the tensor
```



## All below this tbd

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT
