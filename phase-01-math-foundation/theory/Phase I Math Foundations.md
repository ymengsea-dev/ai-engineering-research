## The language that neural networks speak: Linear Algebra and Calulus

- Linear Algebra is the infrastructur (How data is structure and moved).
- Calculus is the engine of learning (How the AI updates it's mistackes).

## Linear Algebra: The Infrastructure

in an LLM, words are turn into arrays of number called vectors, when a model thinks it's just doing massive
ammounts of matrix multiplication.

### Vectors and Matrices

1. **Vector** : A 1D array of numbers. in AI, a vector represents a point in a hight-dimensional space
   e.g., the "meaning" of word

$$\mathbf{v} = [v_1, v_2, v_3]$$

2. **Matric** : A 2D grid of numbers. in LLM, matrices represnt the weight(the knowledge) of the AI.

$$\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$$

### The Dot Product (The Core of Attention)

The dot product multiplies corresponding elements of two vectors and adds them up, resulting in a single number.

$$\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$

noted: The dot product is how AI meansure similarity, The "self-attention" mechanism will use the dot products
to ask "how closely related in the word 'bank' to the word 'river' in this sentence?" A high dot product means
heigh similarity.

### Matrix Multiplication(MatMul)

An operation where you multiply the rows of the first matrix by the columns of the second matrix, element by element, and add the results together.

$\mathbf{Y} = \mathbf{X}\mathbf{W}$

- The Dot Product: The core math engine inside MatMul, it's take two sequence of numbers (like a row and a col), multiplies their corresponding pairs, and sums them up into a single number.

- Data Transformation ($\mathbf{Y} = \mathbf{X}\mathbf{W}$): \* $\mathbf{X}$ is your raw data (features).
  - $\mathbf{W}$ is the model's weights (learned importance/patterns).
  - $\mathbf{Y}$ is the newly transformed data (the model's filtered version of reality used to make decisions).

Example:

- $\mathbf{X}$ (The Input Data): Imagine this is a list of orders from different tables
  - Table 1 orders: 2 Coffees and 1 Cake.

  - Table 2 orders: 1 Coffee and 3 Cakes.

- $\mathbf{W}$ (The Weights): This is the menu price list.
  - Coffee costs $4.
  - Cake costs $5.
- $\mathbf{Y}$ (The Output): This is the final bill for each table.

How the "Dot Product" Works Here:

To find the total bill for Table 1, you match their row (their order) with the price column, multiply the pairs, and add them up:

$$\text{Table 1 Bill} = (2 \text{ Coffees} \times \$4) + (1 \text{ Cake} \times \$5) = 8 + 5 = \$13$$

You do the exact same thing for Table 2:

$$\text{Table 2 Bill} = (1 \text{ Coffee} \times \$4) + (3 \text{ Cakes} \times \$5) = 4 + 15 = \$19$$

### NumPy/PyTorch Broadcasting

- Broadcasting: A mechanism in data science libraries that allows arithmetic operations between tensors (arrays) of different shapes by implicitly "stretching" the smaller tensor to match the dimensions of the larger one.

- How it Works (The Concept)
    - **The Problem** : Computers usually hate doing math between different-sized objects. Trying to add a single row vector of size (1, 3) to a big matrix of size (4, 3) should technically cause an error because the shapes don't match.

    - **The Solution (The Memory Trick)** : Instead of throwing an error, PyTorch and NumPy look at the smaller piece and say, "I can stretch this to match." * The Best Part: It does this virtually. It doesn't actually waste your computer's memory by creating 4 physical copies of the data; it just pretends it did while doing the math at lightning speed.

## Calculus: The Engine of Learning

If linear Algebra is how we pass data through the network to get outpput, Calulus is how we figure out how to charge the weight(The knowledge) when that output is wrong.

### Derivatives and Partial Derivatives

1. **Derivative ($\frac{dy}{dx}$)** : Meansure how much function's output changes when you slightly change it's input.

    > **NOTE** : In AI and Machine Learning, derivatives are used to tweak model weights. If the derivative tells us that increasing a weight slightly will cause the model's error to drop, the AI knows exactly which direction to adjust to learn faster!

2. **Partial Derivative ($\frac{\partial f}{\partial x}$)** : When a function has many inputs (like an LLM with millions of weights), a partial derivative isolates just one weight, treating all other weights as constants, to see how changing that specific weight affects the total error.
    > **NOTE** : Partial derivatives are used to audit every single weight in a model one by one. If the partial derivative for a specific weight tells us that increasing it slightly will cause the model's total error to drop, the AI knows exactly how to tweak that specific weight to learn faster!

### he Chain Rule (The Key to Backpropagation)

An LLM is a deeply nested function: $Output = Layer3(Layer2(Layer1(Input)))$

The Chain Rule states that to find the derivative of a composite function, you multiply the derivatives of its component functions together:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

In AI, this is called Backpropagation. We pass the error backward from the final layer all the way to the first layer, multiplying partial derivatives along the way, so every single weight knows exactly how much it contributed to the mistake.

Example:

Imagine a production line at a toy factory with three workers:

- Worker 1 (layer1) cuts the wood.
- Worker 2 (layer2) paints it.
- Worker 3 (layers) glues the pieces together.

if the final toy turn out completely broken(the error), the factory manager doesn't just scream at the entire room, they go to worker 3 (the final layer) and ask, "How much did your gluing mess this up?" work 3 say, "Well, my gluing was a bit off, but it's mostly because worker 2 gave me a terribly painter, wared piece." So, the manager walks backward to Worker 2, multiplies Worker 3's mistake by Worker 2's mistake, and continues backward to Worker 1. By passing the feedback backwards line-by-line, the manager calculates exactly how a tiny mistake at the very beginning (Worker 1) compounded to ruin the final toy.