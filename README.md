# Notes about Micrograd and NN

## What is a Micrograd
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


## What is a Neural Network

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


