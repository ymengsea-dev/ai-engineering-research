## Build a Linear Regression Engine

### The Goal

Your objective is to build a linear regression engine using pure NumPy. You will implement a model that learns how to fit a straight line to a set of continuous data points by adjusting its weights and bias using Mean Squared Error (MSE) gradients.

### The Blueprint & Instructions

You need to write a Python script that implements a linear modeling system following these precise engineering steps:

1. The Model Equation (Forward Pass):

    Compute the continuous predictions using the linear transformation formula:

    $$\hat{Y} = XW + b$$

2. The Loss Function (compute_mse):

    Implement the Mean Squared Error metric to evaluate how far off your predictions are from the true values ($Y$).

    $$MSE = \frac{1}{m} \sum (\hat{Y} - Y)^2$$

3. The Backward Pass (Vectorized Gradients):

    Calculate the partial derivatives of the MSE loss with respect to the weights ($W$) and bias ($b$). Use matrix operations to compute the gradient across all data points at once.

    $$\mathbf{dW} = \frac{2}{m} \mathbf{X}^T (\mathbf{\hat{Y}} - \mathbf{Y})$$
    
    $$\mathbf{db} = \frac{2}{m} \sum (\mathbf{\hat{Y}} - \mathbf{Y})$$

4. The Update Rule:

    Modify the parameters by moving them down the loss landscape:

    $$W = W - (\text{learning\_rate} \times dW)$$

    $$b = b - (\text{learning\_rate} \times db)$$

### Test Harness & Expected Results

```
import numpy as np

# --- DO NOT ALTER THIS DATA SETUP ---
np.random.seed(42)
# 100 samples, 1 continuous input feature
X = 2 * np.random.rand(100, 1)
# True targets (y = 2X + 5 + random Gaussian noise)
Y = 2 * X + 5 + np.random.randn(100, 1)

# Initialize Weight and Bias exactly like this:
W = np.array([[0.1]])  # Shape (1, 1)
b = np.array([[0.0]])  # Shape (1, 1)

learning_rate = 0.1
epochs = 100
m = len(X)
# ------------------------------------

# TODO: YOUR IMPLEMENTATION HERE
# Write your training loop below...
```

### Expected Outputs to Match:

If your vectorized matrix equations are mapped correctly, your engine will print out these exact values at these epoch milestones:

Epoch 0 (Initial Evaluation):

- Starting Loss should be exactly: 29.1760

Epoch 20:

- Loss should drop down to: 0.9859

- Weight W at this step should be roughly: 2.2858

- Bias b at this step should be roughly: 4.6190

Epoch 100 (Final Iteration):

- Final Loss should be exactly: 0.8055

- Your final weight matrix W should look exactly like this:

    [[1.85903939]]

Your final bias b should look exactly like this:

[[-0.10657989]] (Wait! double check your implementation if your bias matches this: 5.22159821)

Correction Note: Your actual target values should settle at:

W $\approx$ 1.8590

b $\approx$ 5.2216