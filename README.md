# MLP - Multilayer Perceptron

A multilayer perceptron implementation in JavaScript.

## About

This is my implementation of a [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) in JavaScript.
It's comes along with a matrix library to help with the matrix multiplications.
Right now the code is untested and only with basic checks, but I'm still working on it. 
There's a *s* variable commented out in the code, it can be used to measure the error over iterations.
The error should get smaller as the MLP gets trained. The dataset used in the html example was taken from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

## How to use

Let's suppose you have the following data set:

| Height (cm) | Weight (kg) | Class (0-1) |
|-------------|-------------|-------------|
| 180         | 80          | 0           |
| 175         | 67          | 0           |
| 100         | 30          | 1           |
| 120         | 32          | 1           |

0 - adult  
1 - child

You need to process the table to this format:

```
const x = [
	[180, 80],
	[175, 67],
	[100, 30],
	[120, 32]
];

const y = [
	[1,0],
	[1,0],
	[0,1],
	[0,1]
];
```

Note that different from my [perceptron](https://github.com/perceptron) the labels are now [one-hot encoded](https://en.wikipedia.org/wiki/One-hot)

Then just create a new MLP passing the number of inputs, the number of nodes in the hidden layer, the number of outputs,
the learning rate and the number of iterations.


```
const nn = new MLP( x[0].length, x[0].length * 2, 2, 0.03, 500 );
```

Call the fit function

```
nn.fit( x, y );
```

And you're all set to make predictions

```
nn.predict( [178, 70] )
```

There's also a [shuffle](https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks) function that can be used before the training.

```
nn.shuffle( x, y );
```

## Applications

[I trained my neural network to detect when I'm in front of the PC](https://www.youtube.com/watch?v=qcWJdgruG74) (code coming real soon)
