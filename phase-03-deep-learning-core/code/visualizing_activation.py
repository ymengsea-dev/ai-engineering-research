import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

# Let's test three different signal strengths entering our neurons
test_inputs = np.array([-3.5, 0.0, 3.5])

print("--- ACTIVATION FUNCTION TRACER ---")
print(f"Raw Input Signals (Z): {test_inputs}\n")

# 1. Trace Sigmoid
sig_outputs = sigmoid(test_inputs)
print("1. Sigmoid Output (Squashes to 0.0 to 1.0):")
print(f"   Input -3.5 -> {sig_outputs[0]:.4f} (Close to 0)")
print(f"   Input  0.0 -> {sig_outputs[1]:.4f} (Exactly middle probability)")
print(f"   Input  3.5 -> {sig_outputs[2]:.4f} (Close to 1)\n")

# 2. Trace Tanh
tanh_outputs = tanh(test_inputs)
print("2. Tanh Output (Squashes to -1.0 to 1.0):")
print(f"   Input -3.5 -> {tanh_outputs[0]:.4f} (Negative floor)")
print(f"   Input  0.0 -> {tanh_outputs[1]:.4f} (Perfectly zero-centered)")
print(f"   Input  3.5 -> {tanh_outputs[2]:.4f} (Positive ceiling)\n")

# 3. Trace ReLU
relu_outputs = relu(test_inputs)
print("3. ReLU Output (Clamps negatives, leaves positives raw):")
print(f"   Input -3.5 -> {relu_outputs[0]:.4f} (Completely blocked/turned off)")
print(f"   Input  0.0 -> {relu_outputs[1]:.4f} (Zero stays zero)")
print(f"   Input  3.5 -> {relu_outputs[2]:.4f} (Passes through perfectly unchanged)")