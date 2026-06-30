# Welcome to Phase III: Deep Learning Core

This phase we stop looking at flat single layer transformation and start stacking layers to from a deep neural network a "brain"
Untill now, our model could only learn straight lines, By stacking hidden layers and using [Activation Functions](#activation-functions), we unlock the ability to learn complex, non-linear patterns.

## Neural Network Architecture: Stacking Layers

A deep neural network consists of an Input Layer, one or more Hidden Layers, and an Output Layer.

- [The Perceptron](#the-perceptron) / [Artificial Neuron](#artificial-neuron): The core unit. It takes inputs ($X$), multiplies them by weights ($W$), adds a bias ($b$), and passes the result through an activation function ($a$).

- Hidden Layers: Layers of neurons packed between the input and output. Each neuron in a hidden layer extracts different, increasingly abstract features from the data.

### Shape Tracking (Crucial Step!)

When passing data between layers, the columns of the previous layer must match the rows of the next layer.

- If your Input $X$ has a shape of (Batch_Size, Input_Dim)

- And your Hidden Layer weights $W_1$ have a shape of (Input_Dim, Hidden_Dim)

- Then the result $Z_1 = XW_1 + b_1$ will have a shape of (Batch_Size, Hidden_Dim)

**Explain**:

Matrix A shape: $(\text{Rows}, \mathbf{\text{Columns}})$

Matrix B shape: $(\mathbf{\text{Rows}}, \text{Columns})$

The Rule: The Columns of the first matrix must exactly equal the Rows of the second matrix. If they don't match, the computer will crash because the math becomes impossible.

Let's look at the exact shapes from your text and see how they snap together:

1. The Input Data ($X$)
   - Shape: (Batch_Size, Input_Dim)

   - What this means: Imagine you are processing a batch of 4 text messages at the same time (Batch_Size = 4), and each message is represented by 3 numbers (Input_Dim = 3).

   - Your input shape is (4, 3).

2. The Weights Matrix ($W_1$)
   - Shape: (Input_Dim, Hidden_Dim)

   - What this means: This is the layer doing the learning. Its rows must match your input features (3). Let's say this layer outputs 5 new features (Hidden_Dim = 5).

   - Your weights shape is (3, 5).

3. The Multiplication ($X \times W_1$)

   Now we line them up to multiply them:

$$\text{Input } (4, \mathbf{3}) \times \text{Weights } (\mathbf{3}, 5)$$

    See those two 3s in the middle? Because they match, the multiplication is allowed!

What is the Final Shape ($Z_1$)?

To find out what the resulting data looks like after the multiplication, you just look at the outside numbers that are left over:

$$\text{Result Shape} = (\text{First Matrix Rows}, \text{Second Matrix Columns})$$

### Activation Functions: Introducing Non-Linearity

if we just multiply matrices together repeatedly without activation functions, the entire deep network collapses mathematically into a single, simple linear layer.(i.g., Matrix mult after matrix mult is just one giant matrix mult)

Activation functions introduce non-linearity, allowing the network to learn bends, curves, and intricate decision boundaries.

- Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Maps outputs to (0, 1). Great for probabilities, but suffers from vanishing gradients if the network is too deep.

- Tanh: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Maps outputs to (-1, 1). Centered around zero, which often makes optimization faster than Sigmoid.

- ReLU (Rectified Linear Unit): $f(z) = \max(0, z)$. If the input is negative, it outputs 0. If positive, it outputs the input raw. It is computationally fast and is used extensively in modern AI networks.

### Deep Backpropagation

In Phase I, you calculated the derivative for a single layer. In a deep network, backpropagation uses the Chain Rule to pass the error gradient backward through every single hidden layer.

To update the weights of Hidden Layer 1, you must take the error at the Output Layer, multiply it by the gradient of Layer 2's activation, multiply it by Layer 2's weights, and then multiply it by Layer 1's activation gradient.

### Softmax function and Activation function in detail

#### Activation Function (The Spark)

In a biological brain, a neuron recieve singal from other neuron, if the total signal is strong enough, the neuron fires an electrical impulse down the line, if it's weak, it's stay silent.

In AI, an activation function does the exact same thing: it's sit at the output of a neuron layer and decides how much signal to pass to the next layer based on the raw matrix multiplication result 

($Z = XW + b$)

**Why are they mandatory ?**

if we don't use activation functions, stacking 100 layers in a neuron is mathematically identical to having just one layer. why ? because multiplying a matrix by a matrix just results in another matrix. The model remains completely linear.

Activation function introduce non-linearity (bends, clips, and curves). This allow the network to learn in intricate, complex patterns like shapes in images, or syntax rules in programming code.

#### The core Activation function you must know

##### A. Sigmoid

The sigmoid function squashes any raw input value into a strict range between 0 and 1.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Best Used For:** The final layer of a binary classifier (e.g., Is this email spam or not?). Because its output is between 0 and 1, we can interpret the output directly as a probability (like 0.85 = 85% chance of spam).

##### B. Tanh (Hyperbolic Tangent)

Tanh is similar to Sigmoid, but it squashes raw inputs into a range between -1 and 1.

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

- **Best Used For:** Hidden layers in older architectures. Because its output is zero-centered (meaning negative inputs give negative outputs, and positive give positive), it helps keep the data balanced as it flows through the model, making optimization faster than Sigmoid.

##### C. ReLU (Rectified Linear Unit)

ReLU is incredibly simple but revolutionized deep learning. If the input is negative, it outputs 0. If the input is positive, it outputs the input raw.

$$f(z) = \max(0, z)$$

- **Best used for:** Almost all hidden layers in modern deep learning models

- **Why it wins:** It is blindingly fast for a computer to calculate (no complex exponents like $e^{-z}$, just a simple if z > 0 check). More importantly, because the positive side does not flatten out, it completely solves the vanishing gradient problem for deep networks!

#### Softmax function (The LLM Word Picker)

While Sigmoid is great for making yes/no decision, an LLM has a difference problem. When model want to generate the next word, it's has to pick from a dictionary of thousands of words (e.g., "apple", "banana". "computer", "code").

The model out put raw, unconstrained scores for every word in it's vocabulary. These raw scores are called **Logits**.

For example:

- Score for "code": 4.2

- Score for "banana": -1.5

- Score for "river": 1.1

> How do we convert these random, raw scores into a clean probability distribution where all options add up to exactly 1.0 (100%)? We use Softmax.

The Math

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

It does this in two steps:

1. Exponentiation ($e^{z_i}$): It raises $e$ to the power of each score. This turns all negative scores into positive numbers and makes the highest score stand out significantly.

2. Normalization ($\sum$): It divides each exponentiated score by the sum of all exponentiated scores. This forces the final numbers to scale down into percentages that total up to 100%.

---

#### Activation Functions

    Activation Function is a mathematical formular placed at the very end of a modern neuron. it replaces the rigid "0 or 1" threshold of the old perceptron. 
    
    Think of it like the Its job is to take the final combined score of the neuron and transform it before passing it to the next layer.

#### The Perceptron

    The Perceptron is the oldest, simple type of Artifical Neuron, invented back in 1958. The Perceptron is a strict binary decision-maker. It can only answer YES (1) or NO (0).

    real world example is $x_1$: Is the weather nice? (Yes = 1, No = 0)

#### Artificial Neuron

    Actificial Neuron is the foundational building block of all modern AI, it's is a mathematical model inspired by the biological neurons in human brain.

    it work by: takes input data → multiplies them by weights → add them all together → passes the result to an output
