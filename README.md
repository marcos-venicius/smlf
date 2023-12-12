# SMLF

**Small Machine Learning Framework**

This is a small framework to machine learning models.

We have a class `Matrix` and a class `NeuralNetwork`.

# Examples

* [xor](./examples/xor.py)

```console
cd examples

python3 xor.py
```

## Matrix docs

If you wanna print the `Matrix`:

```py
m = Matrix(2, 2)

print(m)
```

If you wanna randomize the `Matrix`:

```py
rand(self) -> None
```

If you wanna fill the `Matrix` with just one value:

```py
fill(self, x: float) -> None
```

If you wanna initialize with specific values:

```py
m1 = Matrix(2, 2, [[0, 1], [0, 1]]) # <-- use the multidimensional array as the third parameter
```

If you wanna set a specific value to a specific location:

```py
set(self, row: int, col: int, x: float) -> None
```

If you wanna sum the `Matrix` with another:

```py
m1 = Matrix(2, 2)
m1.rand()

m2 = Matrix(2, 2)
m2.rand()

m3 = m1.sum(m2) # <-- do this
```

If you wanna apply the sigmoid activation function:

```py
apply_sigmoid(self) -> None
```

If you wanna get a row:

```py
row(row: int) -> Matrix
```

If you wanna get a value a specifc row and column:

```py
at(r: int, c: int) -> float
```

If you wanna multiply the `Matrix` by another:

```py
m1 = Matrix(2, 2)
m2 = Matrix(2, 2)

m1.rand()
m2.rand()

m3 = m1.dot(m2) # <-- do this
```

## NeuralNetwork docs

The neural network expects an architecture like `[2, 2, 1]`, that means:

* 2 inputs
* 2 hidden layers
* 1 output

If you wanna print the `NeuralNetwork`:

```py
nn = NeuralNetwork([2, 2, 1])

print(nn) # <-- do this
```

If you wanna get the `NeuralNetwork` input:

```py
input(self) -> Matrix
```

If you wanna get the `NeuralNetwork` output:

```py
output(self) -> Matrix
```

If you wanna set the input mannually:

```py
set_input(self, input: Matrix) -> None
```

If you wanna randomize the Neural Network (between 0 and 1):

```py
rand(self) -> None
```

If you wanna forward:

```py
forward(self) -> None
```

If you wanna calculate the cost:

```py
cost(self, train_input: Matrix, train_output: Matrix) -> float
```

If you wanna do a finite diff manually (it's made internally):

```py
finite_diff(self, g: NeuralNetwork, epsilon: float, train_input: Matrix, train_output: Matrix) -> None
```

If you wanna make the model learn manually (it's made internally):

```py
learn(self, g: NeuralNetwork, learning_rate: float) -> None
```

If you wanna train the model:

```py
train(self, g: NeuralNetwork, learning_rate: float, epsilon: float, train_input: Matrix, train_output: Matrix, epochs: int) -> None
```
