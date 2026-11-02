
## 🔄 Backpropagation Algorithm

Backpropagation is one of the fundamental algorithms used to train Artificial Neural Networks (ANNs) and Deep Neural Networks (DNNs). It helps neural networks learn by calculating the error between predicted and actual outputs and adjusting weights to minimize that error.

### 📌 Key Concepts

1. **Forward Propagation**
   - Input data passes through the network layer by layer.
   - Each neuron calculates a weighted sum of inputs and applies an activation function.
   - The network generates a predicted output.

2. **Loss / Error Calculation**
   - The predicted output is compared with the actual target.
   - A loss function measures the prediction error.
   - Example: Mean Squared Error (MSE).

3. **Backward Propagation**
   - The error is propagated backward through the network.
   - Derivatives are calculated using the Chain Rule.
   - Gradients determine how much each weight contributes to the error.

4. **Weight Updates**
   - Network weights are adjusted using gradient descent.
   - The learning rate controls the size of each update.

### 🧮 Mathematical Representation

**Gradient Descent Weight Update:**

\[
w_{new} = w_{old} - \alpha \frac{\partial E}{\partial w}
\]

Where:
- `w` = Network weight
- `α` = Learning rate
- `E` = Loss / Error function
- `∂E/∂w` = Gradient of error with respect to weight

### 🔗 Interactive Learning Resource

Learn how backpropagation works through an interactive neural network demonstration by Google for Developers.

👉 **[Explore the Backpropagation Interactive Demo](https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll)**

### 🎯 Learning Outcomes

After studying this concept, you should understand:

- How neural networks process input data.
- How forward propagation generates predictions.
- How loss is calculated.
- How the Chain Rule is used in backpropagation.
- How gradients are calculated.
- How gradient descent updates neural network weights.

---
