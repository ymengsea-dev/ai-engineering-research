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

## Must known concept

### Binary Cross-Entropy (BCE) Loss

When an Ai tries to guess the next word in sentence, it'a doesn't just guess on word. it's looks at it's entire vocabulary and assigns a probability(percentage) to every single word in knows.

if the sentence is "I love writting..." the absolute correct answer might be "code".

- The true answer for "code" is 100%(1.0)
- The true answer for every other word like ("apple", "banana", "car") is 0%(0.0)

**so what is Cross-Entropy ?**

Cross-Entropy is simply the referee's rulebook for punishing the AI when its percentages are wrong.

However, it doesn’t punish linearly. It uses a logarithmic scale, which means it has a massive temper tantrum if the AI is confident and completely wrong.

The Math
For a single example, the Binary Cross-Entropy formula is:

$$L = - \Big( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \Big)$$

- $y$: The actual true label (either 0 or 1).
- $\hat{y}$: The model's predicted probability (a value between 0.0 and 1.0).

**Here is how the referee judges the AI based on that formula:**

#### Case 1: When the true answer is YES ($y = 1$)

The AI is supposed to guess "code".

- AI guesses 99% chance for "code": The referee says, "Great job, you were almost certain!" The penalty (Loss) is almost zero.

- AI guesses 50% chance for "code": The referee says, "Meh, you're unsure." The penalty is medium.

- AI guesses 1% chance for "code": The referee goes crazy. "You were completely confident it WASN'T 'code'!" The penalty (Loss) shoots up toward infinity.

#### Case 2: When the true answer is NO ($y = 0$)

The AI is looking at the word "apple" (which is wrong).

- AI guesses 1% chance for "apple": The referee says, "Correct, it's not apple." Penalty is almost zero.

- AI guesses 99% chance for "apple": The referee says, "Wrong! You were completely sure it was 'apple'!" Penalty shoots up toward infinity.

### Vectorized Gradients

When you train a model on 1000 data points, you could write ```for loop``` to calculate the gradient for the first data point then the second then the third, and add them up.

never do this in AI, python loop are incredibly slow because they execute instruction one by one.

Instead, we use vectorization. we pack all 1000 data points into a single large matrx X, and calculate the gradients for all points simultaneously using simple matrix multiplication operation. Modern computer hardware(CPUs and especially GPUs) are designed to do matrix math in parallel lightning speed.

#### The Non-Vectorized Way (Slow Loops)

```
# Bad practice: Loop over every single sample
for i in range(m):
    prediction = X[i] * W + b
    error = prediction - y[i]
    dW += X[i] * error
dW = (2 / m) * dW
```

#### The Vectorized Way (Fast Matrix Math)

Instead of looking at individuals, we treat the whole batch as a single math statement:

$$\mathbf{dW} = \frac{2}{m} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})$$

By transposing the data matrix ($\mathbf{X}^T$) and multiplying it by the entire error vector ($\mathbf{\hat{y}} - \mathbf{y}$), NumPy automatically computes the dot products for all examples at once.