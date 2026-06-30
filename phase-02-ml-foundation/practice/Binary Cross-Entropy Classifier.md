## Vectorized Binary Cross-Entropy Classifier

### The Goal

Your objective is to build a binary classification engine using pure NumPy that can train a model to separate data points into two distinct classes. You will implement the mathematical formulas yourself, vectorize the operations to avoid loops, and verify your engine against deterministic target values.

### The Blueprint & Instructions

You need to write a Python script that implements a single-layer neural network (a logistic regressor) following these precise steps:

1. The Activation Function (sigmoid):

    Create a function that takes an array of raw scores ($Z$) and maps them between 0 and 1 using: 

    $\sigma(Z) = \frac{1}{1 + e^{-Z}}$.

2. The Loss Function (compute_loss):

    Implement the Binary Cross-Entropy formula. To prevent math crashes ($\log(0) = -\infty$), you must clip your predictions between 1e-15 and 1 - 1e-15 before passing them to the log function.

    $$L = - \frac{1}{m} \sum \Big( Y \log(\hat{Y}) + (1 - Y) \log(1 - \hat{Y}) \Big)$$

3. The Training Loop:

    - Forward Pass: Compute $Z = XW + b$, then pass it through your sigmoid function to get predictions $\hat{Y}$.

    - Backward Pass (Vectorized): Calculate the gradients using matrix math without loops.

    $$\mathbf{dW} = \frac{1}{m} \mathbf{X}^T (\mathbf{\hat{Y}} - \mathbf{Y})$$

    $$\mathbf{db} = \frac{1}{m} \sum (\mathbf{\hat{Y}} - \mathbf{Y})$$

   - Optimization: Update $W$ and $b$ by subtracting the gradient times the learning rate. 

### Test Harness & Expected Results

```
import numpy as np

# --- DO NOT ALTER THIS DATA SETUP ---
np.random.seed(42)
# 100 samples, 2 input features
X = np.random.randn(100, 2)
# True labels (1 if the sum of features is positive, else 0)
Y = (np.sum(X, axis=1, keepdims=True) > 0).astype(float)

# Initialize Weights and Bias exactly like this:
W = np.array([[0.5], [-0.2]])  # Shape (2, 1)
b = np.array([[0.0]])          # Shape (1, 1)

learning_rate = 0.5
epochs = 100
m = len(X)
# ------------------------------------

# TODO: YOUR IMPLEMENTATION HERE
# Write your sigmoid, loss, and training loop below...
```

### Expected Outputs to Match:

Epoch 0 (Initial Evaluation):

Starting Loss should be exactly: 0.6385

Epoch 20:

Loss should drop to roughly: 0.2987

Epoch 100 (Final Iteration):

Final Loss should be exactly: 0.1764

Your final weight matrix W should look exactly like this:

[[2.80554554]

[2.61058692]]

Your final bias b should look exactly like this:

[[-0.10657989]]