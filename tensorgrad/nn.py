import random
from tensorgrad.engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = [[0]]

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, shape, nonlin=True):

        def tensorInitial(rand=True):  # helper to generate an initial tensor for w's
            tensorOut = []
            for x, _ in enumerate(range(shape[0])):
                tensorOut.append([])
                for _ in range(shape[1]):
                    if rand == True:
                        init = random.uniform(-1, 1)
                    else:
                        init = 0
                    tensorOut[x].append(init)
            return tensorOut

        self.w = [Value(tensorInitial()) for _ in range(nin)]
        self.b = [Value(tensorInitial(False)) for _ in range(nin)]
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum(((wi*xi)+bi for wi, xi, bi in zip(self.w, x, self.b)))
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + self.b

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, shape, **kwargs):
        self.neurons = [Neuron(nin, shape, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, shape, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], shape, nonlin=i != len(nouts)-1)
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
