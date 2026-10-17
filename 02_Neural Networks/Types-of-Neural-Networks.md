# 🧠 Types of Neural Networks — Complete Learning Notes

> A structured guide to the major neural-network architectures, how they work, when to use them, their advantages and limitations, and examples in Python.

## 📚 Table of Contents

1. [Introduction](#1--introduction)
2. [How Neural-Network Types Differ](#2--how-neural-network-types-differ)
3. [Perceptron](#3--perceptron)
4. [Feedforward Neural Network (FNN)](#4--feedforward-neural-network-fnn)
5. [Multilayer Perceptron (MLP)](#5--multilayer-perceptron-mlp)
6. [Convolutional Neural Network (CNN)](#6--convolutional-neural-network-cnn)
7. [Recurrent Neural Network (RNN)](#7--recurrent-neural-network-rnn)
8. [Long Short-Term Memory (LSTM)](#8--long-short-term-memory-lstm)
9. [Gated Recurrent Unit (GRU)](#9--gated-recurrent-unit-gru)
10. [Transformer Network](#10--transformer-network)
11. [Autoencoder](#11--autoencoder)
12. [Variational Autoencoder (VAE)](#12--variational-autoencoder-vae)
13. [Generative Adversarial Network (GAN)](#13--generative-adversarial-network-gan)
14. [Diffusion Models](#14--diffusion-models)
15. [Hopfield Network](#15--hopfield-network)
16. [Boltzmann Machine and Restricted Boltzmann Machine](#16--boltzmann-machine-and-restricted-boltzmann-machine)
17. [Self-Organizing Map (SOM)](#17--self-organizing-map-som)
18. [Radial Basis Function Network (RBFN)](#18--radial-basis-function-network-rbfn)
19. [Graph Neural Network (GNN)](#19--graph-neural-network-gnn)
20. [Deep Belief Network (DBN)](#20--deep-belief-network-dbn)
21. [Spiking Neural Network (SNN)](#21--spiking-neural-network-snn)
22. [Capsule Network](#22--capsule-network)
23. [Neural Architecture Comparison](#23--neural-architecture-comparison)
24. [How to Choose a Network](#24--how-to-choose-a-network)
25. [Common Training Concepts](#25--common-training-concepts)
26. [Starter Example: Build an MLP](#26--starter-example-build-an-mlp)
27. [Learning Roadmap](#27--learning-roadmap)
28. [Conclusion](#28--conclusion)
29. [Further Reading](#29--further-reading)

---

## 1. 🌟 Introduction

A **neural network** is a machine-learning model made of connected computational units that learn patterns from data by adjusting trainable parameters such as weights and biases.

Different neural-network architectures are designed to work with different kinds of information:

- **Tabular or fixed-size features:** MLP
- **Images and spatial data:** CNN and vision Transformers
- **Sequences and time series:** RNN, LSTM, GRU, and Transformers
- **Text and language:** Transformers
- **Data generation:** GANs, VAEs, and diffusion models
- **Graph-structured data:** GNNs
- **Unsupervised representation learning:** Autoencoders and related models

There is no single best architecture for every problem. The right choice depends on the data, task, dataset size, computational budget, and evaluation requirements.

## 2. 🔍 How Neural-Network Types Differ

Architectures differ in how they connect units and represent information.

| Design property | What it means |
|---|---|
| Connectivity | Which units can send information to which other units |
| Memory or state | Whether information from earlier inputs affects later outputs |
| Parameter sharing | Whether the same learned filters or transformations are reused |
| Attention | Whether the model learns which parts of the input should influence one another |
| Latent representation | A compact or transformed representation learned from data |
| Training objective | The quantity optimized, such as classification loss, reconstruction loss, or a generative objective |

A network can belong to more than one category. For example, a deep CNN is both a convolutional network and a deep neural network; a Transformer can be used for vision, text, audio, or multimodal tasks.

## 3. 🔹 Perceptron

A perceptron is one of the earliest trainable artificial-neuron models. It computes a weighted sum and applies a threshold-like decision function.

\[
z=\sum_{i=1}^{n} w_i x_i+b
\]

\[
\hat{y}=\begin{cases}1 & z\geq 0\\0 & z<0\end{cases}
\]

### How it works

1. Receive input features.
2. Multiply each feature by its weight.
3. Add the results and bias.
4. Apply a threshold to produce a class.
5. Adjust parameters using a learning rule.

**Best for:** Understanding the foundations of neural networks and simple linearly separable classification.

**Limitation:** A single perceptron cannot learn nonlinear decision boundaries such as XOR.

## 4. ➡️ Feedforward Neural Network (FNN)

In a feedforward network, information moves from the input toward the output without passing through recurrent connections.

```text
Input → Hidden layer(s) → Output
```

### How it works

Each layer transforms the previous layer's output. The model uses the final layer to produce a prediction.

**Applications:** Basic classification, regression, and fixed-size feature processing.

**Advantages:** Simple to understand and implement.

**Limitations:** A basic feedforward network does not naturally maintain sequence memory and may not exploit spatial structure as efficiently as a CNN.

## 5. 🏗️ Multilayer Perceptron (MLP)

An MLP is a feedforward neural network with an input layer, one or more hidden layers, and an output layer. Its layers are commonly fully connected.

For a layer \(l\):

\[
\mathbf{a}^{[l]}=f\left(\mathbf{W}^{[l]}\mathbf{a}^{[l-1]}+\mathbf{b}^{[l]}\right)
\]

### How it works

1. The input features enter the first layer.
2. Dense layers combine information using weights and biases.
3. Nonlinear activations allow the network to learn complex patterns.
4. The output layer produces the prediction.
5. Training updates parameters to reduce the loss.

**Applications:** Tabular classification, regression, and baseline models.

**Advantages:** Flexible and effective for many fixed-size input problems.

**Limitations:** Can need careful tuning; dense layers may use many parameters and do not inherently exploit image locality or graph relationships.

## 6. 🖼️ Convolutional Neural Network (CNN)

CNNs are designed to learn local spatial patterns. They are especially influential in image processing.

### Main components

- **Convolution:** Applies learnable filters across an input.
- **Activation:** Adds nonlinearity, often ReLU.
- **Pooling or strided convolution:** Reduces spatial dimensions.
- **Normalization:** May stabilize training.
- **Dense or task-specific head:** Produces final predictions.

### How it works

Early convolutional layers can learn simple patterns such as edges and textures. Deeper layers may combine these into more complex shapes and object parts. These are learned representations, not manually programmed rules.

**Applications:** Image classification, object detection, segmentation, medical-image research, and video analysis.

**Advantages:** Parameter sharing and local connectivity make CNNs effective for many visual tasks.

**Limitations:** Performance depends on data and design; a CNN may need substantial labeled data and may be sensitive to changes in image conditions.

## 7. 🔁 Recurrent Neural Network (RNN)

RNNs process sequential inputs while maintaining a hidden state that carries information from earlier time steps.

\[
h_t=f(W_xx_t+W_hh_{t-1}+b)
\]

Here, \(x_t\) is the current input and \(h_{t-1}\) is the previous hidden state.

### How it works

1. Read the current element in a sequence.
2. Combine it with the previous hidden state.
3. Compute a new hidden state.
4. Continue through the sequence.
5. Use one or more states to make a prediction.

**Applications:** Time series, sequence labeling, and some speech or language tasks.

**Advantages:** Processes variable-length sequences and maintains a recurrent state.

**Limitations:** Standard RNNs can struggle with long-range dependencies due to vanishing or exploding gradients and are less parallelizable across time steps than Transformers.

## 8. 🧠 Long Short-Term Memory (LSTM)

LSTM is a gated form of recurrent neural network designed to help preserve useful information across longer sequences.

### Main components

- **Forget gate:** Controls which stored information to retain.
- **Input gate:** Controls which new information to store.
- **Cell state:** Carries information through the sequence.
- **Output gate:** Controls what information is exposed as the hidden state.

### How it works

At each time step, gates use learned transformations of the current input and previous hidden state to update the cell state and output.

**Applications:** Time-series forecasting, sequence classification, speech tasks, and legacy or resource-constrained sequence systems.

**Advantages:** Often learns longer dependencies more effectively than a basic RNN.

**Limitations:** Sequential computation can limit parallelism, and Transformers may be preferable for many large-scale sequence tasks.

## 9. 🚦 Gated Recurrent Unit (GRU)

GRU is another gated recurrent architecture. It uses update and reset gates and typically has a simpler state structure than LSTM.

### How it works

- The **update gate** controls how much of the previous state to keep.
- The **reset gate** controls how much previous information to use when forming a candidate state.
- The hidden state is updated using these learned gates.

**Applications:** Sequence modeling and time-series tasks.

**Advantages:** Often has fewer parameters than a comparable LSTM and can be a useful practical alternative.

**Limitations:** Whether it performs better than an LSTM depends on the dataset and task.

## 10. 🔦 Transformer Network

Transformers use attention mechanisms to let input elements interact with other elements in a sequence or representation. The original Transformer architecture was introduced in 2017.

### Core components

- **Token or input embeddings:** Represent input elements numerically.
- **Positional information:** Helps represent order where needed.
- **Self-attention:** Computes how much elements should attend to one another.
- **Feedforward blocks:** Transform each position's representation.
- **Residual connections and normalization:** Help build and train deep stacks.
- **Output head:** Produces task-specific outputs.

### Self-attention intuition

A token can use information from other relevant tokens instead of relying only on a sequential hidden state.

A common scaled dot-product attention formula is:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Here, \(Q\), \(K\), and \(V\) are query, key, and value representations.

### Common Transformer styles

- **Encoder-only:** Often used for representation learning and classification.
- **Decoder-only:** Common in autoregressive text generation.
- **Encoder-decoder:** Useful for tasks that transform one sequence into another.

**Applications:** Language models, translation, summarization, code generation, vision Transformers, audio, and multimodal systems.

**Advantages:** Strong representation learning and efficient parallel processing during many training operations.

**Limitations:** Standard full self-attention can be expensive for long sequences, and large models can require substantial data and compute.

## 11. 🗜️ Autoencoder

An autoencoder learns to reconstruct an input, usually through a compressed or transformed latent representation.

```text
Input → Encoder → Latent representation → Decoder → Reconstruction
```

### How it works

1. The encoder maps input \(x\) to latent representation \(z\).
2. The decoder maps \(z\) to reconstruction \(\hat{x}\).
3. A reconstruction loss compares \(x\) and \(\hat{x}\).
4. Training updates the encoder and decoder to reduce that loss.

**Applications:** Dimensionality reduction, representation learning, denoising, and some anomaly-detection workflows.

**Advantages:** Can learn useful representations without class labels.

**Limitations:** A low reconstruction error does not automatically mean that the latent features are useful for every downstream task. Anomaly detection requires suitable data and threshold validation.

## 12. 🎲 Variational Autoencoder (VAE)

A VAE is a probabilistic generative model related to autoencoders. Rather than mapping an input to only one fixed latent code, its encoder estimates a latent probability distribution.

### How it works

1. The encoder estimates distribution parameters, often a mean and variance.
2. A latent vector is sampled using a differentiable reparameterization technique.
3. The decoder generates a reconstruction from the sampled latent vector.
4. Training combines a reconstruction term with a regularization term that encourages the latent distribution to remain close to a prior.

A common objective is the evidence lower bound (ELBO), expressed as a reconstruction objective plus a Kullback–Leibler divergence term.

**Applications:** Generative modeling, latent-space exploration, and representation learning.

**Advantages:** Provides a structured probabilistic latent space.

**Limitations:** Generated samples can appear blurrier than those from some alternative generative methods, depending on the model and objective.

## 13. 🎭 Generative Adversarial Network (GAN)

A GAN trains two neural networks in competition.

- **Generator:** Creates synthetic samples.
- **Discriminator:** Learns to distinguish real samples from generated ones.

### How it works

1. The generator receives random noise, sometimes with conditioning information.
2. It produces a synthetic sample.
3. The discriminator evaluates real and generated samples.
4. The two networks are trained with an adversarial objective.
5. The generator gradually learns to produce samples that better resemble the training distribution.

**Applications:** Image synthesis, image-to-image translation, and data augmentation research.

**Advantages:** Can generate visually detailed samples.

**Limitations:** Training can be unstable; mode collapse and evaluation difficulties can occur. Generated images may also contain artifacts or misleading details.

## 14. 🌫️ Diffusion Models

Diffusion models are generative models that learn to produce data by reversing a gradual noising process.

### How they work

1. **Forward process:** Noise is gradually added to training data according to a defined schedule.
2. **Learning:** A neural network learns to predict noise or another quantity useful for reversing the process.
3. **Generation:** Start with noise and repeatedly apply learned denoising steps.
4. **Output:** The process produces a sample such as an image or audio segment.

**Applications:** Text-to-image generation, image editing, audio generation, and scientific data generation.

**Advantages:** High-quality generation and flexible conditioning are possible.

**Limitations:** Sampling can require many model evaluations, and results depend on training data, model design, and inference settings.

## 15. 🔄 Hopfield Network

A Hopfield network is a recurrent network with an energy-based interpretation. Classical versions can store patterns as stable states.

### How it works

1. A pattern is represented by network states.
2. Connections define an energy function.
3. The network updates states according to its dynamics.
4. Under suitable conditions, the state settles into a stable pattern or attractor.

**Applications:** Associative memory and theoretical studies of distributed computation.

**Advantages:** Offers a useful model of content-addressable memory.

**Limitations:** Classical networks have storage and retrieval constraints. Modern Hopfield-style models differ from the original formulation.

## 16. 🎚️ Boltzmann Machine and Restricted Boltzmann Machine

Boltzmann machines are stochastic, energy-based neural networks. A **Restricted Boltzmann Machine (RBM)** uses visible and hidden units with no within-layer connections in the standard architecture.

### How an RBM works

1. Visible units represent the observed data.
2. Hidden units learn latent features.
3. The network defines an energy for configurations of visible and hidden units.
4. Training adjusts parameters so that data-like configurations receive more probability.
5. Approximate methods such as contrastive divergence have historically been used for training RBMs.

**Applications:** Historical feature learning, probabilistic modeling, and recommendation research.

**Advantages:** Provides a probabilistic approach to learning latent structure.

**Limitations:** Training and sampling can be difficult, and RBMs are less commonly used in mainstream modern deep-learning pipelines.

## 17. 🗺️ Self-Organizing Map (SOM)

A self-organizing map is an unsupervised neural method that maps high-dimensional observations to a low-dimensional grid while attempting to preserve neighborhood relationships.

### How it works

1. Initialize prototype vectors on a grid.
2. Find the best-matching unit for an input.
3. Update that unit and its neighbors toward the input.
4. Gradually reduce the learning rate and neighborhood size.
5. Inspect the resulting map for structure.

**Applications:** Exploratory data analysis, clustering visualization, and pattern exploration.

**Advantages:** Can make some high-dimensional structures easier to visualize.

**Limitations:** Map quality depends on grid design, scaling, and hyperparameters; a SOM is not a substitute for rigorous clustering evaluation.

## 18. 📍 Radial Basis Function Network (RBFN)

An RBF network often uses a hidden layer of radial basis functions, such as Gaussian functions, centered at selected points.

A Gaussian radial basis function can be written as:

\[
\phi(x)=\exp\left(-\frac{\|x-c\|^2}{2\sigma^2}\right)
\]

Here, \(c\) is a center and \(\sigma\) controls the width.

### How it works

1. Choose or learn centers for the radial basis functions.
2. Measure how close each input is to each center.
3. Use the hidden activations as features.
4. Combine them to produce the output.

**Applications:** Function approximation, interpolation, and some classification or control tasks.

**Advantages:** Can be effective for problems where localized similarity is useful.

**Limitations:** Center selection and scaling can be challenging for high-dimensional data.

## 19. 🕸️ Graph Neural Network (GNN)

GNNs learn from data represented as a graph of nodes and edges. Nodes may represent people, products, molecules, or road intersections; edges represent relationships.

### How message passing works

1. Each node starts with features.
2. It collects or aggregates information from neighboring nodes.
3. A learned update function combines the aggregated messages with its own representation.
4. Several layers allow information from increasingly distant neighborhoods to influence a node.
5. A task-specific head predicts node, edge, or graph properties.

**Applications:** Molecular property prediction, social-network analysis, recommendation, traffic networks, and knowledge graphs.

**Advantages:** Directly incorporates relational structure.

**Limitations:** Oversmoothing, scalability, graph-quality issues, and limited neighborhood information can affect performance.

## 20. 🧱 Deep Belief Network (DBN)

A deep belief network is a generative model built by stacking probabilistic latent-variable layers, historically often trained using restricted Boltzmann machines.

### How it works

1. Learn representations at one layer.
2. Use the learned representation to train the next layer.
3. Stack multiple layers.
4. Historically, fine-tune the model for a downstream task where appropriate.

**Applications:** Historical research on unsupervised pretraining and representation learning.

**Advantages:** Played an important role in the history of deep learning.

**Limitations:** DBNs are less common in modern production workflows than many current architectures, including CNNs and Transformers.

## 21. ⚡ Spiking Neural Network (SNN)

Spiking neural networks represent information through discrete events called spikes, often with neuron states that evolve over time.

### How they work

1. Inputs are encoded as spike events or time-dependent signals.
2. Neuron states evolve according to a dynamical model.
3. When a threshold is reached, a neuron emits a spike.
4. Learning rules adjust synapses or model parameters.

**Applications:** Neuromorphic computing research, event-based sensing, and energy-aware computing research.

**Advantages:** Event-driven computation may be efficient on suitable hardware and for suitable workloads.

**Limitations:** Training, benchmarking, hardware availability, and translating conventional data to spikes can be challenging.

## 22. 🎯 Capsule Network

Capsule networks group neurons into vector-valued units called capsules. The vectors are intended to represent properties of detected entities, such as their presence and pose-related information.

### How they work

1. Lower-level capsules produce vector representations.
2. Routing mechanisms determine how lower-level outputs contribute to higher-level capsules.
3. Higher-level capsules represent more complex entities.
4. A task-specific objective trains the network.

**Applications:** Research into visual recognition and part–whole relationships.

**Advantages:** Explores alternatives to standard scalar activations and pooling.

**Limitations:** Routing can be computationally expensive, and capsule networks have not become a general replacement for CNNs or Transformers.

## 23. 📊 Neural Architecture Comparison

| Type | Data structure | Typical use | Main strength | Main challenge |
|---|---|---|---|---|
| Perceptron | Fixed-size features | Simple linear classification | Very simple | Linear decision boundary |
| MLP | Fixed-size vectors | Tabular prediction | Flexible baseline | No built-in spatial or sequential structure |
| CNN | Grids and images | Vision | Local filters and parameter sharing | May need careful data and augmentation |
| RNN | Sequences | Time series and sequential tasks | Recurrent state | Sequential computation and long-range learning |
| LSTM / GRU | Sequences | Time series and sequence tasks | Gated memory | Less parallelizable than Transformers |
| Transformer | Sequences, patches, multimodal tokens | Language, vision, audio | Attention-based relationships | Compute and memory costs |
| Autoencoder | Many data types | Compression and representation learning | Learns reconstruction features | Reconstruction may not match downstream goals |
| VAE | Many data types | Probabilistic generation | Structured latent distribution | Sample quality and objective trade-offs |
| GAN | Generative data | Synthetic samples | Can create detailed outputs | Training instability |
| Diffusion | Generative data | Image/audio generation | High-quality generation | Sampling cost |
| Hopfield | Pattern states | Associative memory | Energy-based retrieval | Capacity and dynamics constraints |
| RBM | Vectors | Probabilistic feature learning | Latent stochastic structure | Training and sampling complexity |
| SOM | Vectors | Visualization | Neighborhood-preserving map | Sensitive to setup |
| RBFN | Vectors | Approximation | Localized responses | Center selection |
| GNN | Graphs | Relational prediction | Uses graph structure | Scalability and oversmoothing |
| DBN | Vectors / latent layers | Historical representation learning | Layer-wise generative learning | Less commonly used today |
| SNN | Event streams | Neuromorphic research | Spike-based computation | Training and hardware ecosystem |
| Capsule Network | Often images | Part–whole visual modeling | Vector capsules and routing | Complexity and limited adoption |

## 24. 🧭 How to Choose a Network

Use the data structure and task as your starting point, then compare candidate models experimentally.

| Your problem | Reasonable starting point |
|---|---|
| Small structured/tabular dataset | Start with tree-based ML models and an MLP baseline |
| Image classification | CNN or pretrained vision model |
| Object detection | A detector designed for localization, such as a suitable YOLO-family model or another modern detector |
| Time-series forecasting | Establish a statistical or tree-based baseline; then test MLP, LSTM/GRU, or Transformer approaches |
| Text classification | A pretrained Transformer or a simpler baseline |
| Text generation | An appropriate autoregressive language model |
| Image generation | A pretrained diffusion model or another generative architecture |
| Graph prediction | GNN or a graph-aware baseline |
| Compression / reconstruction | Autoencoder |
| Event-based sensor data | Consider SNNs if the hardware and task justify them |

**Practical advice:** Start with a simple baseline, create leakage-safe train/validation/test splits, select models using validation data, and use the held-out test set for final evaluation. Compare quality, latency, memory, robustness, and cost—not accuracy alone.

## 25. 🧰 Common Training Concepts

- **Forward propagation:** Computes predictions from inputs.
- **Loss function:** Measures the discrepancy between predictions and targets or the model's training objective.
- **Backpropagation:** Computes gradients using the chain rule.
- **Optimizer:** Uses gradients to update trainable parameters.
- **Epoch:** One pass through the training data.
- **Batch size:** Number of examples processed in a training step.
- **Learning rate:** Controls update scale.
- **Regularization:** Helps control overfitting, for example through weight decay or dropout.
- **Transfer learning:** Reuses a pretrained model for a related task.
- **Inference:** Runs the trained model to produce outputs.
- **Generalization:** Performance on unseen examples drawn from the intended deployment setting.

## 26. 💻 Starter Example: Build an MLP

The following example uses TensorFlow/Keras to classify samples from the built-in Fashion-MNIST dataset. It demonstrates a basic MLP workflow; it is not intended to be the best architecture for images.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Scale pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Build a simple multilayer perceptron
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train using a validation split from the training data
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
)

# Evaluate on the held-out test set after model selection
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

### What to experiment with

1. Change the number of hidden units.
2. Compare ReLU with another hidden-layer activation.
3. Change the batch size and learning rate.
4. Plot training and validation loss.
5. Compare this MLP with a CNN on the same dataset split.

For a fair comparison, use consistent data splits and evaluation procedures. Tune using validation results rather than repeatedly optimizing against the test set.

## 27. 🛣️ Learning Roadmap

1. Learn Python, NumPy, linear algebra, probability, and basic calculus.
2. Understand perceptrons, MLPs, activations, and loss functions.
3. Implement forward propagation and gradient descent in NumPy.
4. Train small networks using TensorFlow/Keras or PyTorch.
5. Learn evaluation, regularization, and debugging.
6. Study CNNs and build an image-classification project.
7. Study sequence models and Transformers.
8. Explore generative models, GNNs, or another specialty based on your interests.
9. Package a model for inference and document its limitations.

### Suggested mini-projects

- MLP: classify a small tabular dataset.
- CNN: classify images.
- LSTM/GRU: forecast a time series.
- Transformer: classify text using a pretrained model.
- Autoencoder: reconstruct noisy images.
- GNN: predict a graph or molecular property using a suitable dataset.

## 28. 📝 Conclusion

Neural-network architectures differ in how they represent information and learn relationships. MLPs are useful general-purpose baselines; CNNs exploit local spatial patterns; RNNs, LSTMs, and GRUs process sequences through recurrent state; Transformers use attention; autoencoders learn representations; GANs, VAEs, and diffusion models support generation; and GNNs work with graph structure.

The best model is the one that performs reliably on the intended task under realistic constraints. Learn the foundations first, implement small examples, and compare architectures through reproducible experiments.

## 29. 🔗 Further Reading

- [Deep Learning textbook — Goodfellow, Bengio, and Courville](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Attention Is All You Need — Transformer paper](https://arxiv.org/abs/1706.03762)
- [Gradient-Based Learning Applied to Document Recognition — CNN paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [Generative Adversarial Networks — original paper](https://arxiv.org/abs/1406.2661)
- [Auto-Encoding Variational Bayes — VAE paper](https://arxiv.org/abs/1312.6114)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)

---

*Suggested filename: `Types-of-Neural-Networks.md`*
