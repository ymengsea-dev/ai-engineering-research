Imagine a single-neuron AI.

- It takes an input $x = 2.0$
- It has a weight $w = 3.0$.
- The prediction is calculated as: $\hat{y} = x \cdot w$
- The actual correct answer should be $y = 5.0$.
- We measure our error (Loss) using Squared Error: $L = (\hat{y} - y)^2$

**Step 1:** Calculate the current prediction $\hat{y}$ and the current Loss $L$.

**Step 2:** Use the Chain Rule to find the partial derivative of the Loss with respect to the weight ($\frac{\partial L}{\partial w}$).
(Hint: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$)
