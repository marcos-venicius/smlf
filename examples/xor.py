import sys

sys.path.insert(0, '../smlf')

from matrix import Matrix
from neural_network import NeuralNetwork

arch = [2, 2, 1]

nn = NeuralNetwork(arch)
g = NeuralNetwork(arch)

ti = Matrix(4, 2, [[0, 0], [0, 1], [1, 0], [1, 1]])
to = Matrix(4, 1, [[0], [1], [1], [0]])

nn.rand()
g.rand()

nn.train(g, 0.1, 0.1, ti, to, 100000)

print('\nResults: \n')

for row in range(ti.rows):
    input = ti.row(row)

    nn.set_input(input)

    nn.forward()

    output = nn.output()

    print(f'{input.values[0]} ^ {input.values[1]} = {output.values[0]}')
    
