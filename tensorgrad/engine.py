# takes an incoming scalar and converts it to match the shape of this values held data
def convertScalarToMatchTensor(tensor, scalar):
    assert isinstance(scalar, (int, float)
                      ), "only supporting the conversion of int/float for now"
    tensorOut = []
    for x, da in enumerate(tensor):
        tensorOut.append([])
        for dp in da:
            tensorOut[x].append(scalar)
    return tensorOut


class Value:
    """ stores a 2d tensor of scalar values and their gradients """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = []

        # build the gradient data using the incoming data
        for da in data:
            self.grad.append([])
            for dp in da:
                self.grad[len(self.grad)-1].append(0)

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(
            convertScalarToMatchTensor(self.data, other))
        outdata = []

        for x, da in enumerate(self.data):
            outdata.append([])
            for y, dp in enumerate(da):
                outdata[x].append(dp + other.data[x][y])

        out = Value(outdata, (self, other), '+')

        def _backward():
            for x, ga in enumerate(self.grad):
                for y, gp in enumerate(ga):
                    grad = gp + out.grad[x][y]
                    self.grad[x][y] = grad
                    other.grad[x][y] = grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(
            convertScalarToMatchTensor(self.data, other))
        outdata = []
        for x, da in enumerate(self.data):
            outdata.append([])
            for y, dp in enumerate(da):
                outdata[x].append(dp*other.data[x][y])
        out = Value(outdata, (self, other), '*')

        def _backward():
            for x, ga in enumerate(out.grad):
                for y, gp in enumerate(ga):
                    selfGrad = self.grad[x][y] + (other.data[x][y] * gp)
                    self.grad[x][y] = selfGrad
                    otherGrad = other.grad[x][y] + (self.data[x][y] * gp)
                    other.grad[x][y] = otherGrad
        out._backward = _backward

        return out

    def __pow__(self, other):

        outData = []

        for x, da in enumerate(self.data):
            outData.append([])
            for y, dp in enumerate(da):
                assert isinstance(other, (int, float)
                                  ), "only supporting int/float powers for now"
                outData[x].append(dp**other)
        out = Value(outData, (self,), f'**{other}')

        def _backward():
            for x, ga in enumerate(self.grad):
                for y, gp in enumerate(ga):
                    selfGrad = gp + (other * self.data[x][y]
                                     ** (other-1)) * out.grad[x][y]
                    self.grad[x][y] = selfGrad
        out._backward = _backward

        return out

    def relu(self):
        outdata = []
        for x, da in enumerate(self.data):
            outdata.append([])
            for dp in da:
                outdata[x].append(0 if dp < 0 else dp)

        out = Value(outdata, (self,), 'ReLU')

        def _backward():
            for x, ga in enumerate(self.grad):
                for y, gp in enumerate(ga):
                    selfGrad = gp + (out.data[x][y] > 0) * out.grad[x][y]
                    self.grad[x][y] = selfGrad
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
        for x, da in enumerate(self.data):
            self.grad.append([])
            for dp in da:
                self.grad[x].append(1)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

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
