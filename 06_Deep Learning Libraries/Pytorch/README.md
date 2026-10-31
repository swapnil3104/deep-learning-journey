# PyTorch: Complete Learning Notes

> A structured learning resource covering PyTorch fundamentals, model
> development, training, deployment, and advanced techniques.

## 📚 Table of Contents

1.  [Introduction to PyTorch](#1--introduction-to-pytorch)
2.  [Installation and Environment
    Setup](#2--installation-and-environment-setup)
3.  [Tensor Fundamentals](#3--tensor-fundamentals)
4.  [Tensor Operations and
    Broadcasting](#4--tensor-operations-and-broadcasting)
5.  [Autograd and Automatic
    Differentiation](#5--autograd-and-automatic-differentiation)
6.  [Building Neural Networks with
    `torch.nn`](#6--building-neural-networks-with-torchnn)
7.  [Datasets, DataLoaders, and
    Transforms](#7--datasets-dataloaders-and-transforms)
8.  [Training a Model](#8--training-a-model)
9.  [Loss Functions and Optimizers](#9--loss-functions-and-optimizers)
10. [Evaluation, Inference, and
    Checkpoints](#10--evaluation-inference-and-checkpoints)
11. [Computer Vision with PyTorch](#11--computer-vision-with-pytorch)
12. [Natural Language Processing](#12--natural-language-processing)
13. [GPU and Performance
    Optimization](#13--gpu-and-performance-optimization)
14. [Model Saving, Loading, and
    Deployment](#14--model-saving-loading-and-deployment)
15. [Advanced PyTorch Concepts](#15--advanced-pytorch-concepts)
16. [Practical Mini-Project: Image
    Classification](#16--practical-mini-project-image-classification)
17. [Real-World Use Cases](#17--real-world-use-cases)
18. [Advantages and Limitations](#18--advantages-and-limitations)
19. [Best Practices](#19--best-practices)
20. [Common Mistakes and
    Troubleshooting](#20--common-mistakes-and-troubleshooting)
21. [Interview Preparation](#21--interview-preparation)
22. [Quick Revision](#22--quick-revision)

------------------------------------------------------------------------

# 1. 🚀 Introduction to PyTorch

## 1.1 What is PyTorch?

**PyTorch** is an open-source machine learning framework used to build,
train, and deploy machine learning and deep learning models. It provides
tensor computation, automatic differentiation, neural-network building
blocks, and tools for hardware acceleration.

PyTorch is widely used in research and production for computer vision,
natural language processing (NLP), recommendation systems, scientific
computing, generative AI, and reinforcement learning.

## 1.2 Key characteristics

-   **Tensor-based computation:** Multidimensional arrays with CPU and
    accelerator support.
-   **Automatic differentiation:** Computes gradients through operations
    recorded by autograd.
-   **Dynamic computation graphs:** The graph is built as operations
    execute, making Python control flow natural.
-   **Neural-network API:** `torch.nn` provides layers, loss functions,
    and model containers.
-   **Data pipeline tools:** `Dataset` and `DataLoader` support
    efficient input processing.
-   **Ecosystem:** Includes tools and libraries for vision, audio, text,
    distributed training, and deployment.

## 1.3 PyTorch ecosystem

  -----------------------------------------------------------------------
  Component                           Purpose
  ----------------------------------- -----------------------------------
  `torch`                             Tensors, mathematical operations,
                                      devices, and core functionality

  `torch.autograd`                    Automatic differentiation

  `torch.nn`                          Neural-network layers, modules, and
                                      loss functions

  `torch.optim`                       Optimization algorithms

  `torch.utils.data`                  Dataset and DataLoader utilities

  `torchvision`                       Computer vision datasets,
                                      transforms, and models

  `torchaudio`                        Audio processing and datasets

  `torchtext`                         Text-related utilities; check
                                      current project status and
                                      compatibility before adopting

  `torch.distributed`                 Distributed training

  `torch.compile`                     Optional model compilation and
                                      performance optimization
  -----------------------------------------------------------------------

## 1.4 Typical deep-learning workflow

``` mermaid
flowchart TD
    A[Collect data] --> B[Clean and preprocess]
    B --> C[Create Dataset and DataLoader]
    C --> D[Define neural network]
    D --> E[Choose loss and optimizer]
    E --> F[Forward pass]
    F --> G[Calculate loss]
    G --> H[Backpropagate gradients]
    H --> I[Update parameters]
    I --> J{More batches or epochs?}
    J -- Yes --> F
    J -- No --> K[Evaluate model]
    K --> L[Save and deploy]
```

------------------------------------------------------------------------

# 2. 🛠️ Installation and Environment Setup

## 2.1 Install PyTorch

Use the official installation selector at
[pytorch.org](https://pytorch.org/get-started/locally/) to choose a
command compatible with your operating system, Python version, and
accelerator setup.

A basic CPU installation can commonly be installed with:

``` bash
python -m pip install torch torchvision torchaudio
```

For CUDA-enabled or other accelerator builds, use the command generated
by the official selector. The correct package depends on the supported
hardware and software versions.

## 2.2 Verify the installation

``` python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 2.3 Recommended environment

-   Use a virtual environment for each project.
-   Record Python and package versions.
-   Keep the PyTorch and accelerator runtime versions compatible.
-   Use a notebook for exploration and Python modules for reusable
    project code.
-   Fix random seeds when reproducibility is important, while
    remembering that exact determinism can depend on hardware and
    operations.

``` bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
```

------------------------------------------------------------------------

# 3. 🔢 Tensor Fundamentals

## 3.1 What is a tensor?

A **tensor** is a multidimensional data structure. It generalizes
scalars, vectors, and matrices.

  Structure     Rank Example shape
  ----------- ------ ------------------------------------
  Scalar           0 `()`
  Vector           1 `(4,)`
  Matrix           2 `(3, 4)`
  3D tensor        3 `(2, 3, 4)`
  4D tensor        4 `(batch, channels, height, width)`

The **rank** is the number of dimensions. The **shape** gives the size
of each dimension.

## 3.2 Creating tensors

``` python
import torch

# From Python data
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Common initializers
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_values = torch.rand(2, 3)
normal_values = torch.randn(2, 3)
sequence = torch.arange(0, 10, 2)
identity = torch.eye(3)

print(b.shape)   # torch.Size([2, 2])
print(b.ndim)    # 2
print(b.dtype)   # torch.float32
print(b.device)  # cpu, unless moved
```

## 3.3 Important tensor attributes

  Attribute            Meaning
  -------------------- ----------------------------------------------------
  `shape` / `size()`   Dimension sizes
  `ndim`               Number of dimensions
  `dtype`              Data type
  `device`             CPU or accelerator device
  `requires_grad`      Whether autograd tracks operations for gradients
  `is_leaf`            Whether the tensor is a leaf in the autograd graph

## 3.4 Data types

  -----------------------------------------------------------------------
  Type                                Typical use
  ----------------------------------- -----------------------------------
  `torch.float32`                     General neural-network computation

  `torch.float16`                     Reduced-precision accelerator
                                      computation

  `torch.bfloat16`                    Reduced precision with a wider
                                      exponent range than float16

  `torch.float64`                     Higher-precision numerical work

  `torch.int64`                       Class indices and integer indexing

  `torch.bool`                        Boolean masks
  -----------------------------------------------------------------------

``` python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
labels = torch.tensor([0, 1, 2], dtype=torch.long)
mask = torch.tensor([True, False, True])
```

## 3.5 Reshaping and dimensions

``` python
x = torch.arange(12)

matrix = x.reshape(3, 4)
flat = matrix.flatten()
column = matrix.unsqueeze(1)
without_dim = column.squeeze(1)

print(matrix.shape)      # (3, 4)
print(column.shape)      # (3, 1, 4)
print(without_dim.shape) # (3, 4)
```

-   `reshape()` returns a tensor with a requested shape when the number
    of elements matches.
-   `view()` returns a view when the tensor's memory layout permits it.
-   `transpose()` swaps two dimensions.
-   `permute()` reorders dimensions.
-   `unsqueeze()` inserts a dimension of size one.
-   `squeeze()` removes dimensions of size one.
-   `flatten()` combines dimensions into one.

**Important:** `view()` can fail for non-contiguous tensors. `reshape()`
may return a view or make a copy as needed.

## 3.6 Indexing and slicing

``` python
x = torch.tensor([[10, 20, 30], [40, 50, 60]])

print(x[0, 1])   # 20
print(x[1])      # second row
print(x[:, 0])   # first column
print(x[:, 1:])  # columns from index 1 onward

x[0, 0] = 99
```

------------------------------------------------------------------------

# 4. ➗ Tensor Operations and Broadcasting

## 4.1 Arithmetic operations

``` python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)
print(a - b)
print(a * b)  # Element-wise multiplication
print(a / b)
print(a ** 2)
```

## 4.2 Matrix multiplication

``` python
a = torch.randn(2, 3)
b = torch.randn(3, 4)

result = torch.matmul(a, b)
# Equivalent shorthand for matrix multiplication:
result2 = a @ b

print(result.shape)  # (2, 4)
```

Do not confuse `*` (element-wise multiplication) with `@` or
`torch.matmul()` (matrix multiplication).

## 4.3 Useful reduction operations

``` python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(x.sum())
print(x.mean())
print(x.max())
print(x.min())
print(x.argmax())
print(x.sum(dim=0))  # Reduce rows; result for each column
print(x.mean(dim=1)) # Mean of each row
```

## 4.4 Broadcasting

**Broadcasting** allows operations on tensors with compatible shapes
without manually repeating values.

Starting from the last dimension, dimensions are compatible when they
are equal or one of them is `1`. Missing leading dimensions are treated
as `1`.

``` python
a = torch.ones(3, 1)
b = torch.ones(1, 4)

c = a + b
print(c.shape)  # (3, 4)
```

``` mermaid
flowchart LR
    A["Tensor A: (3, 1)"] --> C[Broadcasting]
    B["Tensor B: (1, 4)"] --> C
    C --> D["Output: (3, 4)"]
```

## 4.5 Concatenation and stacking

``` python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

joined = torch.cat([a, b], dim=0)  # Extends an existing dimension
stacked = torch.stack([a, b], dim=0)  # Creates a new dimension

print(joined.shape)   # (4, 2)
print(stacked.shape)  # (2, 2, 2)
```

------------------------------------------------------------------------

# 5. 🧮 Autograd and Automatic Differentiation

## 5.1 What is autograd?

**Autograd** is PyTorch's automatic differentiation engine. It records
operations involving tensors that require gradients and computes
derivatives when `backward()` is called.

For a parameterized model, training uses gradients of a loss function
with respect to model parameters.

## 5.2 Basic gradient example

``` python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

y.backward()

print(y.item())  # 9
print(x.grad)    # 6
```

For the function:

$$
y = x^2
$$

The derivative is:

$$
\frac{dy}{dx} = 2x
$$

At `x = 3`, the gradient is `6`.

## 5.3 Computational graph

``` mermaid
flowchart LR
    A["x = 3"] --> B["y = x²"]
    B --> C["Loss / output"]
    C --> D["backward()"]
    D --> E["dy/dx = 2x"]
```

## 5.4 Gradient accumulation

Gradients accumulate in `.grad` by default. Clear them before each
training update.

``` python
x = torch.tensor(2.0, requires_grad=True)

for _ in range(3):
    y = x ** 2
    y.backward()
    print(x.grad)
    x.grad.zero_()
```

## 5.5 `torch.no_grad()` and inference mode

Use `torch.no_grad()` when gradients are not needed, such as ordinary
validation or inference.

``` python
model_output = None

with torch.no_grad():
    model_output = some_model_input = torch.randn(2, 3)
```

The example above illustrates the context manager; in a real model, call
`model(input_tensor)` inside the block.

`torch.inference_mode()` can provide additional inference optimizations
when its stricter tensor semantics are appropriate:

``` python
with torch.inference_mode():
    predictions = model(input_tensor)
```

## 5.6 Detaching tensors

``` python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()

print(z.requires_grad)  # False
```

Use `detach()` when a tensor should no longer be connected to the
current autograd graph. Avoid unnecessary detaching during training
because it can break gradient flow.

------------------------------------------------------------------------

# 6. 🧠 Building Neural Networks with `torch.nn`

## 6.1 What is `torch.nn`?

`torch.nn` provides reusable building blocks for neural networks,
including layers, activation functions, loss functions, and model
containers.

The usual approach is to define a model as a class that inherits from
`nn.Module`.

## 6.2 Common layers

  -----------------------------------------------------------------------
  Layer                               Purpose
  ----------------------------------- -----------------------------------
  `nn.Linear`                         Fully connected transformation

  `nn.Conv2d`                         2D convolution for image-like
                                      inputs

  `nn.MaxPool2d`                      Spatial downsampling by maximum

  `nn.AvgPool2d`                      Spatial downsampling by average

  `nn.ReLU`                           Rectified linear activation

  `nn.GELU`                           Smooth activation used in many
                                      modern architectures

  `nn.Dropout`                        Randomly disables activations
                                      during training

  `nn.BatchNorm2d`                    Normalizes channel activations
                                      using batch statistics and running
                                      estimates

  `nn.LayerNorm`                      Normalizes across specified feature
                                      dimensions

  `nn.Embedding`                      Maps integer indices to learned
                                      vectors

  `nn.LSTM`                           Recurrent sequence layer

  `nn.MultiheadAttention`             Multi-head attention mechanism
  -----------------------------------------------------------------------

## 6.3 Build a simple neural network

``` python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_features=10, hidden_features=32, classes=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, classes)
        )

    def forward(self, x):
        return self.network(x)

model = SimpleNetwork()
sample = torch.randn(8, 10)
logits = model(sample)

print(logits.shape)  # (8, 3)
```

-   `__init__()` defines layers and submodules.
-   `super().__init__()` initializes the `nn.Module` machinery.
-   `forward()` describes how input flows through the model.
-   Calling `model(x)` invokes PyTorch's module machinery and then
    `forward()`.

## 6.4 Activation functions

  -----------------------------------------------------------------------
  Activation              Formula / behavior      Common use
  ----------------------- ----------------------- -----------------------
  ReLU                    `max(0, x)`             Hidden layers

  Sigmoid                 Maps values to `(0, 1)` Binary probability
                                                  output when paired
                                                  correctly with a loss

  Tanh                    Maps values to          Some recurrent or
                          `(-1, 1)`               bounded representations

  Softmax                 Normalizes scores into  Multiclass inference
                          a probability           
                          distribution            

  GELU                    Smooth nonlinear        Transformers and modern
                          activation              networks

  Leaky ReLU              Small negative slope    Helps avoid completely
                                                  inactive ReLU units
  -----------------------------------------------------------------------

For multiclass classification with `CrossEntropyLoss`, pass raw logits;
do not apply softmax before the loss.

## 6.5 Parameters and model inspection

``` python
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)

total_parameters = sum(p.numel() for p in model.parameters())
trainable_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print("Total:", total_parameters)
print("Trainable:", trainable_parameters)
```

------------------------------------------------------------------------

# 7. 📦 Datasets, DataLoaders, and Transforms

## 7.1 Dataset

A `Dataset` represents a collection of examples. A map-style dataset
commonly implements:

-   `__len__()` --- number of examples.
-   `__getitem__(index)` --- returns one example and its label.

## 7.2 DataLoader

A `DataLoader` batches examples and can shuffle data and load it using
worker processes.

``` python
from torch.utils.data import TensorDataset, DataLoader
import torch

features = torch.randn(1000, 10)
labels = torch.randint(0, 2, (1000,))

dataset = TensorDataset(features, labels)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

for batch_features, batch_labels in loader:
    print(batch_features.shape, batch_labels.shape)
    break
```

## 7.3 Custom Dataset

``` python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

## 7.4 Data pipeline

``` mermaid
flowchart TD
    A[Raw files or records] --> B[Dataset]
    B --> C[Read one sample]
    C --> D[Transform and preprocess]
    D --> E[DataLoader]
    E --> F[Batch and shuffle]
    F --> G[Model]
```

## 7.5 Image transforms

With `torchvision`, transforms can resize, crop, convert, and normalize
images.

``` python
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
```

The normalization values above are commonly associated with
ImageNet-pretrained models. Use the preprocessing expected by the
specific model or dataset.

## 7.6 Data splitting

Separate data into training, validation, and test sets.

  Split        Purpose
  ------------ --------------------------------------------------------
  Training     Fit model parameters
  Validation   Tune hyperparameters and select models
  Test         Final unbiased evaluation after decisions are complete

Avoid data leakage: information from validation or test data must not
influence training or preprocessing decisions that should be learned
only from training data.

------------------------------------------------------------------------

# 8. 🏋️ Training a Model

## 8.1 The training loop

A standard training iteration contains five main operations:

1.  Clear old gradients.
2.  Run the forward pass.
3.  Calculate the loss.
4.  Backpropagate gradients.
5.  Update model parameters.

``` python
for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

## 8.2 Complete training skeleton

``` python
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")
```

## 8.3 Epoch, batch, and iteration

  Term               Meaning
  ------------------ ------------------------------------------------
  Epoch              One complete pass through the training dataset
  Batch              A subset of examples processed together
  Iteration / step   One optimizer update, usually one batch
  Batch size         Number of examples in a batch
  Learning rate      Step-size control for the optimizer

If there are `N` training examples and batch size `B`, the number of
batches per epoch is usually:

$$
\text{Batches per epoch} = \left\lceil \frac{N}{B} \right\rceil
$$

## 8.4 Training versus evaluation mode

``` python
model.train()  # Enables training behavior for modules such as Dropout
model.eval()   # Uses evaluation behavior for modules such as Dropout and BatchNorm
```

`eval()` does not disable autograd. Pair it with `torch.no_grad()` or
`torch.inference_mode()` when gradients are unnecessary.

## 8.5 Validation loop

``` python
model.eval()
validation_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in validation_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        validation_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

accuracy = correct / total if total else 0.0
print("Validation accuracy:", accuracy)
```

------------------------------------------------------------------------

# 9. 🎯 Loss Functions and Optimizers

## 9.1 Loss functions

A loss function measures how far model predictions are from the desired
targets.

  ------------------------------------------------------------------------
  Loss                     Typical task            Expected target
  ------------------------ ----------------------- -----------------------
  `nn.MSELoss`             Regression              Floating-point values

  `nn.L1Loss`              Regression              Floating-point values

  `nn.CrossEntropyLoss`    Single-label multiclass Integer class indices
                           classification          

  `nn.BCEWithLogitsLoss`   Binary or multilabel    Floating-point targets
                           classification          in `[0, 1]`

  `nn.NLLLoss`             Classification from     Integer class indices
                           log-probabilities       

  `nn.SmoothL1Loss`        Robust regression       Floating-point values
  ------------------------------------------------------------------------

### Mean squared error

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

### Binary cross entropy

$$
BCE = -\frac{1}{N}\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a
numerically stable operation. Supply logits, not already-sigmoided
probabilities.

## 9.2 Optimizers

  -----------------------------------------------------------------------
  Optimizer               Key idea                Notes
  ----------------------- ----------------------- -----------------------
  SGD                     Gradient-based updates  Often used with
                                                  momentum and
                                                  learning-rate schedules

  Adam                    Adaptive first- and     Popular general-purpose
                          second-moment estimates choice

  AdamW                   Adam-style updates with Common for transformers
                          decoupled weight decay  and many modern models

  RMSprop                 Adaptive scaling of     Used in some recurrent
                          updates                 and other settings
  -----------------------------------------------------------------------

## 9.3 Optimizer example

``` python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

## 9.4 Learning rate

A learning rate that is too high may cause unstable training; one that
is too low may make convergence very slow.

A simple scheduler:

``` python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

# Call at the appropriate point after each epoch
scheduler.step()
```

Follow the scheduler's documentation for the correct stepping frequency
and order. Some schedulers step per batch; others step per epoch or use
validation metrics.

## 9.5 Gradient descent intuition

For parameter `w`, learning rate `η`, and loss gradient, a basic update
is:

$$
w_{t+1}=w_t-\eta\frac{\partial L}{\partial w_t}
$$

------------------------------------------------------------------------

# 10. 📊 Evaluation, Inference, and Checkpoints

## 10.1 Common metrics

  -----------------------------------------------------------------------
  Metric                  Meaning                 Useful when
  ----------------------- ----------------------- -----------------------
  Accuracy                Fraction of correct     Classes are reasonably
                          predictions             balanced

  Precision               Fraction of predicted   False positives are
                          positives that are      costly
                          correct                 

  Recall                  Fraction of actual      False negatives are
                          positives found         costly

  F1-score                Harmonic mean of        Balancing precision and
                          precision and recall    recall

  MAE                     Mean absolute           Regression
                          prediction error        

  RMSE                    Square root of mean     Regression with
                          squared error           stronger penalty for
                                                  large errors

  AUC-ROC                 Ranking quality across  Binary classification
                          thresholds              ranking
  -----------------------------------------------------------------------

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

Accuracy alone can be misleading for imbalanced datasets. Consider
per-class metrics, confusion matrices, and task-specific costs.

## 10.2 Save a checkpoint

Saving a model's `state_dict` is a flexible and commonly recommended
approach.

``` python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item()
}, "checkpoint.pth")
```

## 10.3 Load a checkpoint

``` python
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

Checkpoint contents and `weights_only` behavior can vary by PyTorch
version and the objects saved. Only load files from trusted sources. If
a checkpoint contains custom Python objects, consult the current
`torch.load` documentation and use an appropriately safe loading
strategy.

## 10.4 Inference

``` python
model.eval()

with torch.inference_mode():
    logits = model(input_tensor.to(device))
    predicted_classes = logits.argmax(dim=1)
```

------------------------------------------------------------------------

# 11. 🖼️ Computer Vision with PyTorch

## 11.1 Image tensor layout

A common image tensor layout is:

$$
(N, C, H, W)
$$

Where:

-   `N` = batch size
-   `C` = channels
-   `H` = height
-   `W` = width

A single RGB image may have shape `(3, H, W)`; a batch commonly has
shape `(N, 3, H, W)`.

## 11.2 Convolutional neural networks

A CNN learns spatial filters that detect patterns such as edges,
textures, shapes, and object parts.

``` mermaid
flowchart LR
    A[Input image] --> B[Convolution]
    B --> C[Activation]
    C --> D[Pooling]
    D --> E[More feature blocks]
    E --> F[Flatten or global pooling]
    F --> G[Classifier]
    G --> H[Class scores]
```

## 11.3 Simple CNN

``` python
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
```

## 11.4 Transfer learning

Transfer learning starts with a model pretrained on a large dataset and
adapts it to a new task.

Typical workflow:

1.  Load a pretrained architecture and its recommended weights.
2.  Apply the associated preprocessing transforms.
3.  Replace the final classification layer.
4.  Freeze the backbone initially if appropriate.
5.  Train the new head.
6.  Optionally unfreeze selected layers and fine-tune with a smaller
    learning rate.

``` python
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 5)

# Only the new classifier is trainable at this stage.
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

preprocess = weights.transforms()
```

------------------------------------------------------------------------

# 12. 📝 Natural Language Processing

## 12.1 NLP data representation

Text is converted into numerical representations before it can be
processed by neural networks.

Common stages:

1.  Text normalization when appropriate.
2.  Tokenization.
3.  Mapping tokens to integer IDs.
4.  Padding or truncation to form batches.
5.  Embedding lookup.
6.  Sequence model or transformer processing.
7.  Prediction and evaluation.

## 12.2 Embeddings

An embedding maps a discrete token ID to a learned dense vector.

``` python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
token_ids = torch.tensor([[12, 45, 78], [9, 10, 0]])

vectors = embedding(token_ids)
print(vectors.shape)  # (2, 3, 128)
```

## 12.3 Attention

Attention lets a model compute context-dependent combinations of
representations. In scaled dot-product attention, queries, keys, and
values are combined as:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where `d_k` is the key dimension.

## 12.4 Transformer overview

``` mermaid
flowchart TD
    A[Input text] --> B[Tokenizer]
    B --> C[Token IDs]
    C --> D[Token embeddings]
    D --> E[Positional information]
    E --> F[Transformer blocks]
    F --> G[Task-specific head]
    G --> H[Prediction]
```

A transformer block commonly includes multi-head self-attention,
residual connections, normalization, and a feed-forward network. Exact
ordering and details depend on the architecture.

------------------------------------------------------------------------

# 13. ⚡ GPU and Performance Optimization

## 13.1 Select a device

``` python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

inputs = inputs.to(device)
targets = targets.to(device)
```

Other accelerators may be available depending on the PyTorch build and
platform. Check current device documentation for MPS, XPU, and other
backends.

## 13.2 Mixed precision

Mixed precision can improve throughput and reduce memory use on
supported hardware. A typical CUDA pattern is:

``` python
scaler = torch.amp.GradScaler("cuda")

for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Use mixed precision only on a supported device and validate numerical
stability. Current APIs may differ by PyTorch version and device.

## 13.3 Performance practices

-   Keep tensors and model parameters on the same device.
-   Avoid frequent CPU-GPU transfers inside the training loop.
-   Use suitable batch sizes.
-   Use `pin_memory=True` and `non_blocking=True` where they help a CUDA
    input pipeline.
-   Profile before optimizing.
-   Consider `torch.compile(model)` where supported and beneficial.
-   Use efficient data loading and preprocessing.
-   Avoid synchronizing the GPU unnecessarily, for example by repeatedly
    converting CUDA values to Python scalars.

## 13.4 DataLoader performance example

``` python
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

for inputs, targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
```

Choose `num_workers` based on the operating system, data source, and
workload. On some systems, more workers can make performance worse. In
scripts using multiprocessing on Windows and some other environments,
protect the entry point with `if __name__ == "__main__":`.

------------------------------------------------------------------------

# 14. 💾 Model Saving, Loading, and Deployment

## 14.1 Save model weights

``` python
torch.save(model.state_dict(), "model_weights.pth")
```

## 14.2 Load model weights

Recreate the same architecture before loading its parameters:

``` python
model = SimpleNetwork()
state = torch.load(
    "model_weights.pth",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state)
model.eval()
```

## 14.3 Deployment options

  -----------------------------------------------------------------------
  Option                              Purpose
  ----------------------------------- -----------------------------------
  Python service                      Integrate a model into an API or
                                      application

  TorchScript                         Legacy/compatibility-oriented model
                                      representation; assess current
                                      support for the target workflow

  `torch.export`                      Capture a PyTorch program for
                                      downstream deployment workflows

  ONNX export                         Interoperability with supported
                                      runtimes

  Mobile or edge runtime              Run models on constrained devices,
                                      subject to current ecosystem
                                      support

  Cloud inference                     Serve models using scalable
                                      infrastructure
  -----------------------------------------------------------------------

Deployment requires more than saving weights: define preprocessing,
input/output schemas, versioning, monitoring, latency targets, and safe
model-loading practices.

------------------------------------------------------------------------

# 15. 🧪 Advanced PyTorch Concepts

## 15.1 Learning-rate schedulers

Schedulers adjust the learning rate during training.

Examples include:

-   `StepLR`
-   `MultiStepLR`
-   `CosineAnnealingLR`
-   `ExponentialLR`
-   `ReduceLROnPlateau`
-   `OneCycleLR`

Choose a schedule based on the training regime and follow its required
stepping pattern.

## 15.2 Gradient clipping

Gradient clipping limits gradient magnitude and can help stabilize
training, especially in some recurrent or transformer training settings.

``` python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 15.3 Weight initialization

Good initialization can improve optimization.

``` python
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(initialize_weights)
```

Do not blindly overwrite pretrained weights with a generic initializer.

## 15.4 Freezing and unfreezing layers

``` python
for parameter in model.backbone.parameters():
    parameter.requires_grad = False

# Later, selectively unfreeze:
for parameter in model.backbone.layer4.parameters():
    parameter.requires_grad = True
```

When unfreezing parameters, ensure the optimizer includes them. You may
need to add a parameter group or recreate the optimizer.

## 15.5 Distributed training

Distributed training uses multiple GPUs or machines to accelerate
training or handle larger workloads.

Common approaches:

-   **DistributedDataParallel (DDP):** A widely used multi-process
    approach that synchronizes gradients.
-   **Fully Sharded Data Parallel (FSDP):** Shards model state to
    support larger models.
-   **Tensor parallelism:** Splits tensor operations across devices.
-   **Pipeline parallelism:** Splits model layers into stages across
    devices.

``` mermaid
flowchart TD
    A[Training dataset] --> B[Distributed sampler]
    B --> C1[Worker process / GPU 1]
    B --> C2[Worker process / GPU 2]
    C1 --> D1[Model replica 1]
    C2 --> D2[Model replica 2]
    D1 --> E[Gradient synchronization]
    D2 --> E
    E --> F[Consistent parameter updates]
```

For multi-GPU training, use the official distributed tutorials and
account for process initialization, samplers, synchronization,
checkpointing, and failure handling.

## 15.6 Custom autograd functions

Most operations should use built-in PyTorch operators. A custom
`torch.autograd.Function` is appropriate when implementing a specialized
operation whose backward behavior must be defined.

``` python
class MultiplyByThree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor * 3

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 3

x = torch.tensor(2.0, requires_grad=True)
y = MultiplyByThree.apply(x)
y.backward()
print(x.grad)  # 3
```

## 15.7 Compilation

`torch.compile` can optimize supported models by compiling and
specializing execution.

``` python
compiled_model = torch.compile(model)
```

Compilation may introduce startup overhead, graph breaks, or
compatibility constraints. Benchmark the compiled model on
representative workloads rather than assuming it is always faster.

## 15.8 Reproducibility

``` python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

Exact reproducibility is not guaranteed across all hardware, software
versions, kernels, and execution environments. Consult PyTorch's
reproducibility guidance when strict determinism is required.

------------------------------------------------------------------------

# 16. 🧩 Practical Mini-Project: Image Classification

## 16.1 Objective

Build a small neural network that classifies images into categories.
This example uses a synthetic tensor dataset so the code can run without
downloading a dataset. It demonstrates the complete training mechanics;
it is not intended to produce a meaningful real-world classifier.

## 16.2 Project workflow

``` mermaid
flowchart TD
    A[Create synthetic images and labels] --> B[TensorDataset]
    B --> C[DataLoader]
    C --> D[Define CNN]
    D --> E[Choose CrossEntropyLoss]
    E --> F[Train for multiple epochs]
    F --> G[Evaluate]
    G --> H[Save weights]
    H --> I[Run inference]
```

## 16.3 Complete runnable example

``` python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. Reproducibility and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Synthetic RGB images: 3 channels, 32 x 32 pixels
num_samples = 600
num_classes = 3
images = torch.randn(num_samples, 3, 32, 32)
labels = torch.randint(0, num_classes, (num_samples,))

dataset = TensorDataset(images, labels)

train_size = 500
validation_size = num_samples - train_size
train_dataset, validation_dataset = random_split(
    dataset, [train_size, validation_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)

# 3. Define model
class ImageClassifier(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

model = ImageClassifier().to(device)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 5. Training
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- training loss: {total_loss / len(train_loader):.4f}"
    )

# 6. Validation
model.eval()
correct = 0
total = 0

with torch.inference_mode():
    for batch_images, batch_labels in validation_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_images)
        predictions = logits.argmax(dim=1)

        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

print("Validation accuracy:", correct / total)

# 7. Save model weights
torch.save(model.state_dict(), "image_classifier.pth")

# 8. Inference on one sample
model.eval()
sample = images[0].unsqueeze(0).to(device)

with torch.inference_mode():
    prediction = model(sample).argmax(dim=1).item()

print("Predicted class:", prediction)
```

## 16.4 How to turn this into a real project

1.  Replace synthetic data with a labeled image dataset.
2.  Use a proper train/validation/test split.
3.  Apply training augmentation and model-specific validation
    preprocessing.
4.  Consider transfer learning with a pretrained model.
5.  Track metrics and inspect a confusion matrix.
6.  Save the best validation checkpoint.
7.  Test on data not used during model selection.
8.  Package preprocessing and inference into a repeatable application.

------------------------------------------------------------------------

# 17. 🌍 Real-World Use Cases

  -----------------------------------------------------------------------
  Domain                  Example                 Typical model family
  ----------------------- ----------------------- -----------------------
  Healthcare research     Medical-image           CNNs, U-Nets,
                          segmentation or         transformers
                          classification          

  Retail                  Product recognition and CNNs, sequence models
                          demand forecasting      

  Banking                 Fraud detection and     MLPs, graph neural
                          risk modeling           networks

  Manufacturing           Visual defect detection CNNs, anomaly-detection
                                                  models

  Autonomous systems      Object detection and    CNNs, vision
                          scene understanding     transformers

  NLP                     Text classification,    Transformers
                          translation,            
                          summarization           

  Recommendation          Ranking products or     Embeddings, deep
                          content                 ranking models

  Speech                  Speech recognition and  Convolutional and
                          audio classification    transformer models

  Science                 Simulation surrogates   Neural networks,
                          and physical-system     differentiable models
                          modeling                

  Generative AI           Image, audio, and text  Diffusion models,
                          generation              transformers
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 18. ⚖️ Advantages and Limitations

## 18.1 Advantages

-   Python-friendly and expressive API.
-   Dynamic graphs support flexible model logic.
-   Strong autograd support.
-   Broad set of neural-network components.
-   CPU and accelerator execution.
-   Large research and developer ecosystem.
-   Suitable for prototyping and production workflows.
-   Supports distributed training and modern optimization tools.

## 18.2 Limitations

-   Deep-learning workloads can require substantial GPU memory and
    compute.
-   Performance tuning can be complex.
-   Package, driver, and accelerator compatibility can be challenging.
-   Dynamic execution may require extra work for some deployment
    targets.
-   Reproducibility can be affected by nondeterministic operations and
    hardware differences.
-   Large projects need disciplined data, experiment, and model
    management.
-   A framework does not automatically solve data quality, bias,
    privacy, security, or model-monitoring problems.

## 18.3 PyTorch vs. TensorFlow

  -----------------------------------------------------------------------
  Aspect                  PyTorch                 TensorFlow
  ----------------------- ----------------------- -----------------------
  Programming style       Python-first, eager     Eager execution with
                          execution by default    graph and compilation
                                                  workflows

  Model API               `nn.Module`             Keras and lower-level
                                                  APIs

  Differentiation         `autograd`              `GradientTape` and
                                                  automatic
                                                  differentiation

  Deployment              Multiple export and     TensorFlow Serving,
                          serving paths           LiteRT and other
                                                  ecosystem tools

  Research workflow       Widely used in research Also widely used in
                                                  research and production

  Best choice             Depends on team,        Depends on team,
                          deployment target, and  deployment target, and
                          ecosystem               ecosystem
  -----------------------------------------------------------------------

Neither framework is universally better. Evaluate the actual model,
hardware, deployment constraints, team experience, and maintenance
requirements.

------------------------------------------------------------------------

# 19. ✅ Best Practices

## 19.1 Code and model design

-   Build models from small, testable modules.
-   Use descriptive names for layers, tensors, and functions.
-   Keep preprocessing consistent between training and inference.
-   Verify tensor shapes at model boundaries.
-   Prefer built-in operations over custom kernels unless necessary.
-   Keep configuration separate from training logic.

## 19.2 Training

-   Start with a simple baseline.
-   Track training and validation metrics.
-   Use a validation set for model selection.
-   Save the best checkpoint, not just the final epoch.
-   Clear gradients every iteration.
-   Choose loss functions that match the target format.
-   Tune learning rate and batch size systematically.
-   Use early stopping when appropriate.
-   Inspect class imbalance and data leakage.

## 19.3 Performance and reliability

-   Profile before optimizing.
-   Move data and model to the intended device.
-   Use inference mode for inference when appropriate.
-   Use mixed precision only after checking hardware support and
    numerical behavior.
-   Record package versions, random seeds, data versions, and
    configuration.
-   Load only trusted checkpoints.
-   Monitor production data quality, latency, drift, and failure cases.

------------------------------------------------------------------------

# 20. 🧯 Common Mistakes and Troubleshooting

  -------------------------------------------------------------------------
  Mistake                   Why it happens          Solution
  ------------------------- ----------------------- -----------------------
  Forgetting                Gradients accumulate    Clear gradients before
  `optimizer.zero_grad()`                           each update

  Calling                   Parameters update       Backpropagate first,
  `optimizer.step()` before without current         then step
  `backward()`              gradients               

  Applying softmax before   Loss expects logits     Pass raw logits
  `CrossEntropyLoss`                                

  Wrong target dtype        Classification targets  Use `torch.long` for
                            often need integer      class indices
                            indices                 

  Shape mismatch            Batch, channel, or      Print and verify tensor
                            feature dimensions      shapes
                            differ                  

  CPU/GPU mismatch          Model and data are on   Move both to the same
                            different devices       device

  Forgetting `model.eval()` Dropout and BatchNorm   Call `eval()` for
                            remain in training      evaluation and
                            behavior                inference

  Using `eval()` as a       Evaluation mode does    Use `no_grad()` or
  substitute for no-grad    not disable autograd    `inference_mode()` too

  Reusing stale gradients   `.grad` retains         Zero gradients every
                            previous values         iteration

  Overfitting               Model memorizes         Improve data,
                            training examples       regularize, augment, or
                                                    simplify

  Data leakage              Validation/test         Split data before
                            information enters      fitting preprocessing
                            training                or selecting models

  Out-of-memory error       Model, activations, or  Reduce batch size, use
                            batch exceed device     mixed precision, or
                            memory                  optimize memory

  Incorrect normalization   Input distribution      Follow the dataset or
                            differs from model      pretrained model's
                            expectation             preprocessing

  Saving only architecture  Weights require         Recreate the same
  assumptions               matching model          architecture and load
                            structure               its state dict
  -------------------------------------------------------------------------

## 20.1 Debugging checklist

``` python
print("Input:", inputs.shape, inputs.dtype, inputs.device)
print("Target:", targets.shape, targets.dtype, targets.device)
print("Model device:", next(model.parameters()).device)
```

For a classification batch, verify:

-   Logits shape is `(batch_size, number_of_classes)`.
-   Class-index targets shape is `(batch_size,)`.
-   Targets use `torch.long` for `CrossEntropyLoss`.
-   Model and tensors use compatible devices and dtypes.

------------------------------------------------------------------------

# 21. 🎤 Interview Preparation

## 21.1 Common interview questions

**1. What is PyTorch?**

An open-source deep-learning framework providing tensors, automatic
differentiation, neural-network modules, optimizers, and accelerator
support.

**2. What is a tensor?**

A multidimensional array with a data type and device, supporting
mathematical operations and optional gradient tracking.

**3. What is autograd?**

PyTorch's automatic differentiation engine, which records relevant
tensor operations and calculates gradients during backpropagation.

**4. What is the difference between `view()` and `reshape()`?**

`view()` requires a compatible memory layout and returns a view.
`reshape()` returns a view when possible or a copy when necessary.

**5. What is the difference between `model.train()` and
`model.eval()`?**

They set module behavior for training or evaluation, affecting modules
such as Dropout and BatchNorm. They do not directly enable or disable
gradient calculation.

**6. What does `loss.backward()` do?**

It computes gradients of the loss with respect to leaf tensors requiring
gradients and accumulates them in their `.grad` fields.

**7. Why do we use `optimizer.zero_grad()`?**

Gradients accumulate by default. Clearing them prevents unintended
accumulation across optimizer steps.

**8. What is the difference between `torch.no_grad()` and
`torch.inference_mode()`?**

Both avoid autograd tracking for inference-like work. Inference mode is
more restrictive and can enable additional optimizations.

**9. What is a DataLoader?**

A utility that batches data from a dataset and supports shuffling,
multiprocessing, and other input-pipeline features.

**10. What is transfer learning?**

Reusing a pretrained model and adapting some or all of its parameters to
a related task.

**11. Why use CrossEntropyLoss with logits?**

It combines the required log-softmax and negative log-likelihood
computation in a numerically stable way.

**12. What is overfitting?**

When a model performs well on training data but generalizes poorly to
unseen data.

**13. What is gradient clipping?**

A technique that limits gradient magnitude or norm to help prevent
excessively large updates.

**14. What is the purpose of `state_dict()`?**

It provides a mapping of a module's parameters and persistent buffers,
commonly used for saving and restoring model state.

**15. What is the difference between SGD and Adam?**

SGD uses gradient-based updates, optionally with momentum. Adam adapts
updates using estimates of gradient first and second moments.

## 21.2 Important terms to remember

  -----------------------------------------------------------------------
  Term                                Short definition
  ----------------------------------- -----------------------------------
  Tensor                              Multidimensional array

  Autograd                            Automatic gradient calculation

  Forward pass                        Compute predictions from inputs

  Backward pass                       Compute gradients

  Loss                                Measure of prediction error

  Optimizer                           Updates model parameters

  Epoch                               Full pass over training data

  Batch                               Group of samples

  Learning rate                       Optimizer step-size parameter

  Logits                              Raw, unnormalized model scores

  Overfitting                         Poor generalization after fitting
                                      training data

  Regularization                      Techniques that discourage
                                      undesirable model complexity

  Embedding                           Learned vector representation

  Checkpoint                          Saved training or model state

  Fine-tuning                         Further training a pretrained model

  Inference                           Using a trained model to make
                                      predictions
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 22. ⚡ Quick Revision

## 22.1 Core training sequence

``` python
model.train()

for inputs, targets in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

## 22.2 Essential commands

  -----------------------------------------------------------------------
  Command                             Use
  ----------------------------------- -----------------------------------
  `torch.tensor(data)`                Create a tensor

  `torch.zeros(shape)`                Create zeros

  `torch.ones(shape)`                 Create ones

  `torch.randn(shape)`                Sample standard normal values

  `torch.arange(start, end)`          Create a sequence

  `x.reshape(shape)`                  Reshape a tensor

  `x.view(shape)`                     Create a compatible view

  `x.unsqueeze(dim)`                  Add a dimension

  `x.squeeze(dim)`                    Remove a size-one dimension

  `torch.cat(tensors, dim)`           Concatenate

  `torch.stack(tensors, dim)`         Stack along a new dimension

  `torch.matmul(a, b)`                Matrix multiplication

  `loss.backward()`                   Calculate gradients

  `optimizer.zero_grad()`             Clear gradients

  `optimizer.step()`                  Update parameters

  `model.train()`                     Set training mode

  `model.eval()`                      Set evaluation mode

  `torch.no_grad()`                   Disable gradient tracking in a
                                      block

  `torch.inference_mode()`            Run inference with stricter
                                      optimized semantics

  `model.state_dict()`                Get model state

  `torch.save()`                      Save objects or state

  `torch.load()`                      Load a saved file
  -----------------------------------------------------------------------

## 22.3 Key formulas

Mean squared error:

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

Gradient descent:

$$
w_{t+1}=w_t-\eta\nabla_w L
$$

Precision:

$$
Precision=\frac{TP}{TP+FP}
$$

Recall:

$$
Recall=\frac{TP}{TP+FN}
$$

F1-score:

$$
F1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}
$$

Scaled dot-product attention:

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 22.4 End-to-end learning roadmap

``` mermaid
flowchart TD
    A["1. Python and NumPy basics"] --> B["2. Tensor creation and operations"]
    B --> C["3. Autograd and gradients"]
    C --> D["4. nn.Module and neural networks"]
    D --> E["5. Dataset and DataLoader"]
    E --> F["6. Loss functions and optimizers"]
    F --> G["7. Training and validation"]
    G --> H["8. CNNs and transfer learning"]
    H --> I["9. NLP and transformers"]
    I --> J["10. GPU and performance"]
    J --> K["11. Checkpointing and deployment"]
    K --> L["12. Distributed and advanced training"]
```

## 22.5 Final revision checklist

-   [ ] Explain tensors, shape, rank, dtype, and device.
-   [ ] Create, index, reshape, and broadcast tensors.
-   [ ] Explain autograd and gradient accumulation.
-   [ ] Build a model using `nn.Module`.
-   [ ] Select a loss function for a task.
-   [ ] Explain optimizer and learning-rate behavior.
-   [ ] Implement a complete training loop.
-   [ ] Separate training, validation, and test data.
-   [ ] Use `train()`, `eval()`, and inference mode correctly.
-   [ ] Move models and tensors to the same device.
-   [ ] Save and load a model checkpoint safely.
-   [ ] Explain CNNs, embeddings, attention, and transfer learning.
-   [ ] Identify overfitting, data leakage, and common shape errors.
-   [ ] Profile and optimize before scaling to multiple devices.

## 22.6 Recommended learning resources

-   Official documentation: [PyTorch
    Documentation](https://pytorch.org/docs/stable/index.html)
-   Tutorials: [PyTorch Tutorials](https://pytorch.org/tutorials/)
-   Installation: [Get
    Started](https://pytorch.org/get-started/locally/)
-   Model ecosystem:
    [TorchVision](https://pytorch.org/vision/stable/index.html)
-   Distributed training: [Distributed
    Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)

------------------------------------------------------------------------

**End of PyTorch Notes**
