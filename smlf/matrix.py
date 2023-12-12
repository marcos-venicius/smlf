import math
from random import random

class Matrix:
    def __init__(self, rows, cols, initial_value=None):
        assert rows > 0, "rows cannot be less than or equal to 0"
        assert cols > 0, "cols cannot be less than or equal to 0"

        self.rows = rows
        self.cols = cols
        self.stride = cols
        self.values = [] if initial_value else [0 for _ in range(self.rows * self.cols)]
        self.print_padding = 0

        if initial_value:
            assert len(initial_value) == rows, f"the initial value should have {rows} rows"

            for r in range(rows):
                assert len(initial_value[r]) == cols, f"the number of cols of your initial value should be equal to {cols}"

                for v in initial_value[r]:
                    self.values.append(v)


    def __sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def rand(self):
        for i in range(self.rows * self.cols):
            self.values[i] = random()

    def fill(self, x):
        for i in range(self.rows * self.cols):
            self.values[i] = x

    def set(self, row, col, x):
        assert row >= 0, "the row should be greater than or equal to 0"
        assert col >= 0, "the col should be greater than or equal to 0"
        assert row <= self.rows, "the row should be less than or equal to the total matrix rows"
        assert col <= self.cols, "the col should be less than or equal to the total matrix cols"

        self.values[row * self.stride + col] = x

    def sum(self, matrix):
        assert matrix.rows == self.rows, "to sum matrices the number of rows should be equal"
        assert matrix.cols == self.cols, "to sum matrices the number of cols should be equal"

        m = []

        for i in range(self.rows * self.cols):
            m.append(self.values[i] + matrix.values[i])

        nm = Matrix(self.rows, self.cols)

        nm.values = m

        return nm

    def apply_sigmoid(self):
        for i in range(self.rows * self.cols):
            self.values[i] = self.__sigmoid(self.values[i])

    def row(self, row):
        assert row >= 0, "the row should be greater than or equal to 0"
        assert row <= self.rows - 1, "the row should be less than or equal to the number of rows that this matrix has - 1"

        m = Matrix(1, self.cols);

        m.values = self.values[row * self.stride:row * self.stride + self.stride]

        return m

    def at(self, r, c):
        return self.values[self.stride * r + c]

    def dot(self, matrix):
        assert self.cols == matrix.rows, "to multiply matrices the number of cols of the first one should be equal to the number of rows of the second one"

        def at(s, r, c):
            return s * r + c

        a = self
        b = matrix
        m = Matrix(a.rows, b.cols)

        for row in range(a.rows):
            for col in range(b.cols):
                for i in range(a.cols):
                    m.values[at(m.stride, row, col)] += a.values[at(a.stride, row, i)] * b.values[at(b.stride, i, col)] 

        return m

    def __str__(self):
        padding = self.print_padding
        self.print_padding = 0
        string = (' ' * padding) + "[\n"

        stride = 0

        n = self.cols * self.rows

        for i in range(n):
            ident = stride == 0

            if ident:
                string += '    '

            string += (' ' * padding) + str(self.values[i])

            if stride == self.stride - 1 and i < n - 1:
                stride = 0
                string += '\n'
            else:
                stride += 1

        string += '\n' + (' ' * padding) + "]"

        return string
