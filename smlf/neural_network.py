from tqdm import tqdm
from matrix import Matrix

class NeuralNetwork:
    def __init__(self, arch):
        assert len(arch) > 0, "the arch should be creater than 0"
        
        self.layers_count = len(arch) - 1

        self._ws = [None for i in range(self.layers_count)]
        self._bs = [None for i in range(self.layers_count)]
        self._as = [Matrix(1, arch[0])] + [None for i in range(self.layers_count)]

        for i in range(1, len(arch)):
            self._ws[i - 1] = Matrix(self._as[i - 1].cols, arch[i])
            self._bs[i - 1] = Matrix(1, arch[i])
            self._as[i] = Matrix(1, arch[i])

    def output(self):
        return self._as[self.layers_count]

    def input(self):
        return self._as[0]

    def __str__(self):
        string = 'NN = [\n'

        for i in range(self.layers_count):
            w = self._ws[i]
            w.print_padding = 4
            string += f'    w{i} ({w.rows}x{w.cols}) ='
            string += str(w)
            string += '\n'

            b = self._bs[i]
            b.print_padding = 4

            string += f'    b{i} ({b.rows}x{b.cols}) ='
            string += str(b)
            string += '\n'

        string += ']\n'

        return string

    def set_input(self, input: Matrix):
        for i in range(input.rows * input.cols):
            self.input().values[i] = input.values[i]

    def rand(self):
        for w in self._ws:
            w.rand()

        for b in self._bs:
            b.rand()

    def forward(self):
        for i in range(self.layers_count):
            self._as[i + 1] = self._as[i].dot(self._ws[i])
            self._as[i + 1] = self._as[i + 1].sum(self._bs[i])
            self._as[i + 1].apply_sigmoid()

    def cost(self, train_input: Matrix, train_output: Matrix):
        assert train_input.rows == train_output.rows, "the number of rows of both train data should be equal"
        assert train_output.cols == self.output().cols, "the number of cols of the train output must be equal to the number of cols of the neural network output"

        c, n = 0, train_input.rows

        for i in range(n):
            x = train_input.row(i)
            y = train_output.row(i)

            # copy given input to the NN input
            for j in range(x.rows * x.cols):
                self.input().values[j] = x.values[j]

            self.forward()

            for j in range(train_output.cols):
                d = self.output().at(0, j) - y.at(0, j)

                c += d * d

        return c / n

    def finite_diff(self, g, epsilon: float, train_input: Matrix, train_output: Matrix):
        saved = 0.0;

        c = self.cost(train_input, train_output)

        for li in range(self.layers_count):
            w = self._ws[li]
            b = self._bs[li]

            for i in range(w.rows * w.cols):
                saved = w.values[i]

                w.values[i] += epsilon
                g._ws[li].values[i] = (self.cost(train_input, train_output) - c) / epsilon
                w.values[i] = saved

            for i in range(b.rows * b.cols):
                saved = b.values[i]

                b.values[i] += epsilon
                g._bs[li].values[i] = (self.cost(train_input, train_output) - c) / epsilon
                b.values[i] = saved

    def learn(self, g, learning_rate: float):
        for li in range(self.layers_count):
            w = self._ws[li]
            b = self._bs[li]

            for i in range(w.rows * w.cols):
                w.values[i] -= learning_rate * g._ws[li].values[i]

            for i in range(b.rows * b.cols):
                b.values[i] -= learning_rate * g._bs[li].values[i]

    def train(self, g, learning_rate: float, epsilon: float, train_input: Matrix, train_output: Matrix, epochs: int):
        print('training...\n')

        for i in tqdm(range(epochs)):
            self.finite_diff(g, epsilon, train_input, train_output)
            self.learn(g, learning_rate)

        print('\ndone.')

