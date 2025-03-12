# Notes about Micrograd and NN

## Micrograd
- A minified Autograd engine that implements backpropagation, an algorithm to efficitely evaluate a gradient of some type of loss function with respect to the weights of a neural network
- We can tune the weights of a neural network to minimize the loss function
- It is a scalar value autograd engine rather than n-dimensional tensors used in production grade neural networks which are arrays of these scalars that allow us to take advantage of parallelism within computers (used purely for efficiency)
- An autograd a more powerful version of a micrograd due to how it deals with data, scalars vs tensors
- Both and autograd and micrograd can be used to power a neural network 

### Micrograd vs Autograd
- Micrograd
    - Scalar Operations: Works with individual numbers (scalars). This means that every operation, even a simple matrix multiplication, needs to be broken down into individual scalar operations     
    - Computational Graph: Micrograd builds a computational graph to track all the operations. This graph can become very large and complex for larger networks, leading to memory and performance issues.   
    - Limited Optimization: Micrograd is designed for simplicity, not performance. It doesn't have the advanced optimizations that libraries like PyTorch and TensorFlow have for handling large-scale computations.
- Autograd
    - Tensor Operations: Work with tensors (multi-dimensional arrays). This allows for vectorized operations, which are much faster and more efficient, especially on GPUs.   
    - Optimized Code: These libraries are highly optimized for tensor operations, taking advantage of specialized hardware and algorithms.
    - Automatic Differentiation: They automatically compute gradients for complex functions, including those involving tensors.   
    - Scalability: Designed to handle large datasets and complex models, making them suitable for real-world applications.
- Example of Difference for Handling Matrix Multiplication
    - Micrograd would implement matrix multiplication by breaking it down into a series of scalar multiplicates and additions --> it would loop through the rows and columns performing the dot product for each element
    - Autograd engines would perform matrix multiplicates as a single optimized operation by leveraging efficient algorithms (implemented in lower-level languages) that are designed specifically for matrix operations --> this is a lot faster


## Neural Networks

### Structure
- Input Layer: Receives the initial data (e.g., pixel values of an image).   
- Hidden Layers: One or more layers where the actual processing happens. Neurons in these layers learn to extract features from the input data.   
- Output Layer: Produces the final result (e.g., classification of an image).

### Function
- Forward Pass: Data flows through the network from the input layer to the output layer. Each neuron performs a simple calculation on its inputs and passes the result to the next layer.   
- Learning: The network learns by adjusting the connections (weights) between neurons. This adjustment is based on the difference between the network's output and the desired output.   
- Backpropagation: This is where "grad" comes in. Backpropagation is an algorithm that calculates the gradients (derivatives) of the error with respect to each weight in the network. These gradients tell us how much each weight needs to be adjusted to reduce the error.   
- Optimization: An optimization algorithm (like gradient descent) uses the gradients to update the weights, iteratively improving the network's performance.

### Gradients
- A gradient is a more general derivative that can be used for n-variable functions
- The gradient of a function tells you the rate of change of that function with respect to multiple variables
- It's a vector that points in the direction of the greatest increase of the function
- Neural networks learn by adjusting their weights to minimize the error
- The gradient of the error function tells the network how to change each weight to reduce the error 
- It's like the compass that guides the network towards better performance

### How Gradients Power Neural Networks
- Gradients provide the direction for adjusting the network's parameters (weights and biases) to minimize the error (minimal error --> more accurate model)
    - Gradient as Direction: The gradient at a particular point tells you the direction of the steepest ascent. To find the lowest point, you need to go in the opposite direction of the gradient.
    - Backpropagation as the Guide: Backpropagation efficiently calculates these "direction indicators" (gradients) for all the weights in the network.   
    - Optimization as the Journey: Optimization algorithms use these gradients to iteratively update the weights, taking small steps in the direction that reduces the error, eventually leading the network to a state where it performs well on the given task.
- Gradients act as a compass, guiding the neural network towards better performance by indicating how to adjust its parameters

### Neurons
- A neuron in a neural network is a computational unit inspired by biological neurons
- It takes in inputs, processes them, and produces an output that is passed to other neurons in the network
- There are 6 components in the structure of a neuron

**1) Inputs (Synapses)**
- Neurons receive multiple inputs, often represented as x1, x2, ..., xn
- The inputs can come from raw data (from the very first input layer) or from neurons in previous layers

**2) Weights**
- Each input is assigned a weight (w) that determines its importance
- The weight is a trainable parameter that gets updated during training
- Higher weights amplify the signal; lower weights diminish it

**3) Bias**
- A bias term is added to the weighted sum of inputs
- Helps shift the activation function, allowing better fitting of complex patterns
- Bias ensures the neuron can activate even if all inputs are zero (it's like a constant value in y=mx+b)

**4) Summation Function (Weighted sum)**
- The neuron calculates the weighted sum of inputs: 
```
                            z = w1x1 + w2x2 + ... + wnxn + b
```
- This value, z, determines the neuron's pre-activation state (the value calculated by the neuron before the activation function is applied)

**5) Activation Function f(z)**
- Transforms the weighted sum into an output value
- Introduces non-linearity, allowing the network to model complex patterns
- Common functions
    - ReLU (max(0, z)) - Helps prevent vanishing gradients
        - The vanishing gradient problem occurs during the training of deep neural networks, where gradients used to update the network's weights become extremely small as they propagate backward through the layers
        - Activation functions with gradients that are consistently less than 1 (like sigmoid or tanh in their saturation regions) can cause these multiplied gradients to become exponentially smaller, eventually "vanishing"
        - This makes it difficult for the earlier layers to learn effectively
        - ReLU's linear nature in the positive region, with a constant gradient of 1, allows gradients to flow more freely through the network, significantly reducing the risk of vanishing gradients
    - Sigmoid (1/(1+e^-z)) - Maps output to (0, 1) for probability estimation
    - Tanh - Maps output to (-1, 1) often used in hidden layers

**6) Output**
- The final output of the neuron, which can be passed to the next layer of neurons
- In classification tasks (e.g. finding the probability the inputted image is car vs bus), the last layer often uses a softmax function to normalize outputs into probabilities which can be interpreted as percentages of certainty for categories
    - The softmax function is a mathematical function that takes a vector of real numbers (the raw outputs of the last layer) and transforms them into a probability distribution
    - Characteristics
        - Probabilities: It ensures that all output values are between 0 and 1, representing probabilities
        - Sum to 1: It guarantees that the sum of all output probabilities is equal to 1
        - This allows us to treat the outputs as the likelihood that the input belongs to each class

**Analogy**
- A neuron is like a judge making a decision
    - The inputs are pieces of evidence
    - The weights determine how important each piece of evidence is
    - The bias is like a base level of leniency or strictness
    - The summation is like adding up all the evidence
    - The activation function decides whether the judge rules in favor or against based on the total evidence
    - The output is the final verdict

**Parameters**
- The parameters of a neural network are the weights and biases, they're essentially the learnable parts of the model
- They determine how the inputs are transformed
- During training, the learning algorithm adjusts these weights and paremeters (parameters) to reduce error on a task
- These parameters then transform the inputs fed into the neuron based on previous findings/trainings

### Forward Propogation & Inference
- Forward propagation is the process of passing input data through a neural network to compute an output (prediction)
- It involves matrix multiplications, applying activation functions, and moving data from the input layer to the output layer'
- Forward propagation is essentially what inference is
    - During training: Forward propagation is followed by backpropagation, where gradients are calculated and weights are updated
    - During inference: Only forward propagation happens—weights are fixed, and the network simply computes predictions
- When you use ChatGPT, Gemini, or any AI model in production, you're interacting with inference

### Backpropogation
- Backpropagation (backward propagation of errors) is the process used to train neural networks by adjusting their weights to minimize error
- It uses the chain rule of calculus to compute gradients and update weights through gradient descent (steepest derivative for multi-variable functions)
- When dealing with multiple layers we:
    - Compute loss at the output layer
    - Propagate gradients backward through hidden layers (in between input and ouput)
    - Adjust weights layer by layer
- Backpropagation is how neural networks "learn" by adjusting weights to reduce prediction error

**Example Walkthrough**

With this example

```
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh()

Neuron Workflow Tree
├── Inputs
│   ├── x1 = Value(2.0, label='x1')
│   └── x2 = Value(0.0, label='x2')
│
├── Parameters (Neuron’s Learnable Weights)
│   ├── w1 = Value(-3.0, label='w1')
│   ├── w2 = Value(1.0, label='w2')
│   └── b  = Value(6.8813, label='b')
│
├── Computation Steps
│   ├── Step 1: Weighted Multiplication
│   │   ├── x1w1 = x1 * w1 = 2.0 * (-3.0) = -6.0
│   │   └── x2w2 = x2 * w2 = 0.0 * 1.0 = 0.0
│   │
│   ├── Step 2: Summing the Products
│   │   └── x1w1x2w2 = x1w1 + x2w2 = -6.0 + 0.0 = -6.0
│   │
│   ├── Step 3: Adding the Bias
│   │   └── n = x1w1x2w2 + b = -6.0 + 6.8813 = 0.8813
│   │
│   └── Step 4: Non-Linear Activation
│       └── o = tanh(n) = tanh(0.8813) ≈ 0.707
│
└── Output
    └── o (the activated output of the neuron)
```

When we execute the `draw_dot(o)` function, we'll essentially be doing DFS starting from the top level node o.

```
          x2=0.0          w2=1.0
            │               │
            ▼               ▼
            \             /
  (multiply) *  <------> * (multiply)
            /             \
           ▼               ▼
     x2*w2=0.0       x1*w1=-6.0
           │               │
           ▼               ▼
            \             /
      (add)  +  <----->  +   (add)
            /             \
           ▼               ▼
        x1*w1 + x2*w2 = -6.0
                   │
                   ▼
                b=6.8813
                   │
                   ▼
                   +   (add)
                   │
                   ▼
                n=0.8813
                   │
                   ▼
                (tanh)
                   │
                   ▼
               o=0.7071
```

Here's how we do backpropogation and fill in the gradients of each node you see above:
- Starting with o, we ask ourselves what is the derivative of o with respect to o, it's just 1.0
- Now we back propogate through the tanh, `tanh(n)`.
    - To do this, we need to know the local derivative of tanh
    - We know that o = tanh(n), then we need to find do/dn
    - derivative of tanh(n) is 1-tanh(n)^2 --> do/dn = 1 - o**2 (o = tanh(n))
    - This evalutes to 0.5
- Now we go through `(x1\*w1 + x2\*w2) + b`, a plus node is just a distributor of a gradient so the gradient of the previous `tanh` step will flow into each of the elements of the plus equally
    - Note that all a gradient does is see how much a node affects another node
    - When you make slight changes in an expression that adds things together, the result differs by that slight change you made
    - In the context of backpropogation, you want to find the change that you're making on the end output, so in this case we distribute the 0.5 gradient of the n node to the `x1\*w1 + x2\*w2` and `b=6.8814` terms making both of their gradients 0.5 as well
- With `x1\*w1 + x2\*w2`, same thing as above, it's a plus node so we distribute 0.5 equally to each
- For `x2\*w2`, this is a multiplication node so the local derivative of one term is in the other term
    - The gradient of `x2` will be `w2*(x2\*w2)'`
    - This is just the property of chain rule
    - The reason we get that is because the derivative of `x2\*w2` with respect to `x2` is `w2`
    - By the chain rule, you multiply the upstream gradient `x2\*w2` by `w2`
    - Thus the result is gradients `w2 = 0` and `x2 = 0.5` (note that this makes sense because the value of `x2` is 0, so incrementing `w2` by small amounts will essentially do nothing to the end result)
- Repeat the above step for the others