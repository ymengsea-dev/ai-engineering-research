# Welcome to Phase II: ML Fundamentals

Now that we have raw mathematical primitives, we are going to use them to build model that are actually learn from data. instead of jumping straight into automated libraries, we are going to write Linear Regression, Classifier, and Gradient Descent using pure Numpy.

Why Numpy? because in PyTorch, magic happends behind the scences. In NumPy, you are the engine. Writing these algorithms from scratch will give you an intuitive understanding of weights, biases, and data shapes that will make coding your models significantly easier.

## NumPy for Numerical Computing

NumPy is the backbone of scientific compution in Python, it's allow us to perform vectorized operations on array, which mean we can process thousands of data points simultaneously without writing slow python for loop.

Key operations to master:

- **np.array.shape**: Always know your dimensions.
- **np.dot(A, B) or A @ B**: Matrix multiplication.
- **Vectorization**: Performing an operation ( like '+' or '*' ) on a whole array at once.

## Linear Regression: Predicting Continus Values

Linear Regression is the process of fitting a straight line to a set of data points. if you want your AI to predict a continous number (like the next token probability value or score), you are using regression principle.

The equation of a singl prediction is:

$$\hat{y} = XW + b$$

- $X$: Your input data matrix (Shape: [Num Examples, Features]).
- $W$: Your weights/parameters (Shape: [Features, 1]).
- $b$: Your bias term (a scalar offset that lets the line move up and down).

### The Error Metric: Mean Squared Error (MSE)

To know how bad our line is, we calculate the average squared distance between our prediction ($\hat{y}$)  and the actual target  ($y$):

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

## Classification: Making Decisions

While regression predictions a continuous number, Classification predicts a category or class (e.g., "is this next word a noun or a verb?", or "is this image a cat or a dog?"). Because LLM ultimately classify which token out of a vocabulary of thousands should come next, this is a crucial step.

To turn a continuous regression line ($\hat{y}$) into a probability between 0 and 1, we pass it through the **Sigmoid Function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

If the output is $> 0.5$, we classify it as 1; otherwise, it's 0.

## Gradient Descent: Teaching the Model

Gradient Descent is the optimization algorithm used to train almost all modern AI models. It uses the calculus we learned in Phase I to iteratively tweak the weights ($W$) and bias ($b$) to lower the total loss.

The update rule is:

$$W = W - (\text{Learning Rate} \times \nabla_W L)$$

The Learning Rate ($\alpha$) is a tiny step size (like 0.01). If it's too big, the model overshoots and breaks; if it's too small, the model takes forever to learn.
