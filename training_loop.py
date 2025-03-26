from nn import MLP
from typing import *

# Sample problem with sample NN
def setup_nn():
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    n(x)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    ypred = [n(x) for x in xs]

def training_loop(loops: int, ypred: List[float], xs: List[List[float]], ys: List[float], n: MLP, learning_rate: float = 0.01):
    # draw_dot(n(x))

    for k in range(loops):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
        
        # backward pass
        for p in n.parameters():
            p.grad = 0.0 # reset to 0 otherwise the grads of previous steps will accumlate and give massive/incorrect step sizes
        loss.backward()
        
        # update
        for p in n.parameters():
            p.data += -learning_rate * p.grad
        
        # print(k, loss.data)