## What Just Happened Under the Hood?

1. The Forward Pass & Computation Graph

   when PyTorch executes y_pred_tensor = x_tensor \* w_tensor, it doesn't just calculate 6.0. Under the hood, it builds a dynamic Computation Graph. It remembers that y_pred_tensor was created by multiplying x and w.

2. The Backward Pass (.backward())

   When you call loss_tensor.backward(), PyTorch reads this graph in reverse:
   - It looks at the final loss formula and calculates its derivative.
   - It multiplies it by the derivative of the multiplication step (the Chain Rule).
   - It stores the final answer directly in the .grad property of any tensor where requires_grad=True.

3. Interpreting the Gradient (4.0)

   What does a gradient of 4.0 actually mean?It means: "Right now, if you increase the weight $w$ by a tiny amount (say, $0.01$), the loss will increase 4 times as fast (by roughly $0.04$)."Because our goal is to minimize loss, we want to move the weight in the opposite direction of the gradient. If the gradient is positive, we subtract it from the weight to make our AI smarter!
