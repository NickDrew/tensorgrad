

class Value:
    """ stores a list of scalar values and their gradients """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = []
        for dp in data:
            self.grad.append(0)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        outdata = []

        for x, dp in enumerate(self.data):
            outdata.append(dp + other.data[x])

        out = Value(outdata, (self, other), '+')

        def _backward():
            for x, gp in enumerate(self.grad):
                grad = gp + out.grad[x]
                self.grad[x] = grad
                other.grad[x] = grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        outdata = []
        for x, dp in enumerate(self.data):
            outdata.append(dp*other.data[x])
        out = Value(outdata, (self, other), '*')

        def _backward():
            for x, gp in enumerate(out.grad):
                selfGrad = self.grad[x] + (other.data[x] * gp)
                self.grad[x] = selfGrad
                otherGrad = other.grad[x] + (self.data[x] * gp)
                other.grad[x] = otherGrad
        out._backward = _backward

        return out

    def __pow__(self, other):

        outData = []

        for x, dp in enumerate(self.data):
            assert isinstance(other[x], (int, float)
                              ), "only supporting int/float powers for now"
            outData.append(dp**other[x])

        out = Value(outData, (self,), f'**{other}')

        def _backward():
            for x, gp in enumerate(self.grad):
                selfGrad = gp + (other[x] * self.data[x]
                                 ** (other[x]-1)) * out.grad[x]
                self.grad[x] = selfGrad
        out._backward = _backward

        return out

    def relu(self):
        outdata = []
        for dp in self.data:
            outdata.append(0 if dp < 0 else dp)

        out = Value(outdata, (self,), 'ReLU')

        def _backward():
            for x, gp in enumerate(self.grad):
                selfGrad = gp + (out.data[x] > 0) * out.grad[x]
                self.grad[x] = selfGrad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = []
        for dp in self.data:
            self.grad.append(1)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        reverser = []
        for dg in self.data:
            reverser.append(-1)
        return self * reverser

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
