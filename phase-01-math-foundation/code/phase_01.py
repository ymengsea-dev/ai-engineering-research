import torch

print("--- APPROACH 1: PURE PYTHON (MANUAL MATH) ---")
# 1. Inputs and initial values
x = 2.0
w = 3.0
y = 5.0

# Forward pass: calculate prediction and error (Loss)
y_pred = x * w                    # 2.0 * 3.0 = 6.0
loss = (y_pred - y) ** 2          # (6.0 - 5.0)^2 = 1.0

print(f"Prediction (y_pred): {y_pred}")
print(f"Loss (L): {loss}")

# Backward pass: applying the Chain Rule manually
# dL/dy_pred = 2 * (y_pred - y)
# dy_pred/dw = x
# dL/dw = dL/dy_pred * dy_pred/dw
dL_dypred = 2 * (y_pred - y)      # 2 * (6.0 - 5.0) = 2.0
dypred_dw = x                     # 2.0
dL_dw = dL_dypred * dypred_dw     # 2.0 * 2.0 = 4.0

print(f"Manual Gradient (dL/dw): {dL_dw}\n")


print("--- APPROACH 2: PYTORCH (AUTOMATIC AUTOGRAD) ---")
# We specify requires_grad=True to tell PyTorch to track operations on this variable
x_tensor = torch.tensor(2.0)
w_tensor = torch.tensor(3.0, requires_grad=True)
y_tensor = torch.tensor(5.0)

# Forward pass (looks exactly like regular math)
y_pred_tensor = x_tensor * w_tensor
loss_tensor = (y_pred_tensor - y_tensor) ** 2

# Backward pass: PyTorch traverses the computation graph backward
loss_tensor.backward()

print(f"PyTorch Prediction: {y_pred_tensor.item()}")
print(f"PyTorch Loss: {loss_tensor.item()}")
print(f"PyTorch Gradient (w_tensor.grad): {w_tensor.grad.item()}")