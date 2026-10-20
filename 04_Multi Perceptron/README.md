# 🧠 Multi-Layer Perceptron (MLP) --- Complete Learning Notes

## 📚 Table of Contents

1.  [Introduction](#1--introduction)
2.  [What Is a Perceptron?](#2--what-is-a-perceptron)
3.  [What Is a Multi-Layer
    Perceptron?](#3--what-is-a-multi-layer-perceptron)
4.  [MLP Architecture](#4--mlp-architecture)
5.  [Core Terminology](#5--core-terminology)
6.  [Mathematical Foundation](#6--mathematical-foundation)
7.  [Activation Functions](#7--activation-functions)
8.  [Forward Propagation](#8--forward-propagation)
9.  [Loss Functions](#9--loss-functions)
10. [Backpropagation and Gradient
    Descent](#10--backpropagation-and-gradient-descent)
11. [Training Workflow](#11--training-workflow)
12. [Implementing an MLP with
    Scikit-learn](#12--implementing-an-mlp-with-scikit-learn)
13. [Implementing an MLP with
    TensorFlow/Keras](#13--implementing-an-mlp-with-tensorflowkeras)
14. [Practical Mini-Project:
    Classification](#14--practical-mini-project-classification)
15. [Use Cases and Real-World
    Examples](#15--use-cases-and-real-world-examples)
16. [Advantages and Limitations](#16--advantages-and-limitations)
17. [MLP vs Other Models](#17--mlp-vs-other-models)
18. [Hyperparameter Tuning](#18--hyperparameter-tuning)
19. [Regularization and
    Generalization](#19--regularization-and-generalization)
20. [Advanced Concepts](#20--advanced-concepts)
21. [Best Practices](#21--best-practices)
22. [Common Mistakes and
    Troubleshooting](#22--common-mistakes-and-troubleshooting)
23. [Interview Questions and
    Answers](#23--interview-questions-and-answers)
24. [Quick Revision](#24--quick-revision)

------------------------------------------------------------------------

## 1. 🎯 Introduction

A **Multi-Layer Perceptron (MLP)** is a type of feed-forward artificial
neural network made of interconnected neurons arranged in an input
layer, one or more hidden layers, and an output layer. It learns
patterns by adjusting weights and biases during training.

MLPs are commonly used for classification and regression on structured
or tabular data. They can learn nonlinear relationships when nonlinear
activation functions are used in hidden layers.

### Learning objectives

By the end of these notes, you should be able to:

-   Explain a perceptron and how it differs from an MLP.
-   Identify the layers and parameters in an MLP.
-   Understand forward propagation, loss, backpropagation, and gradient
    descent.
-   Select suitable activation and loss functions.
-   Train and evaluate an MLP in Python.
-   Diagnose overfitting, underfitting, and unstable training.

### High-level workflow

``` mermaid
flowchart TD
    A[Collect labeled data] --> B[Clean and preprocess]
    B --> C[Split train, validation, test]
    C --> D[Choose MLP architecture]
    D --> E[Forward propagation]
    E --> F[Calculate loss]
    F --> G[Backpropagation]
    G --> H[Update weights and biases]
    H --> I{Converged or stopping rule met?}
    I -- No --> E
    I -- Yes --> J[Evaluate on held-out test set]
    J --> K[Deploy and monitor]
```

------------------------------------------------------------------------

## 2. 🔹 What Is a Perceptron?

A **perceptron** is a simple artificial neuron that computes a weighted
sum of its inputs, adds a bias, and applies an activation function.

### 2.1 Perceptron equation

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
\hat{y} = f(z)
$$

Where:

-   $$x_i$$: input feature.
-   $$w_i$$: weight associated with the feature.
-   $$b$$: bias.
-   $$z$$: weighted sum before activation.
-   $$f$$: activation function.
-   $$\hat{y}$$: predicted output.

### 2.2 Simple example

Suppose a model estimates whether a student passes using study hours and
attendance.

  Input                    Value   Weight
  ---------------------- ------- --------
  Study hours (scaled)       0.8      0.7
  Attendance (scaled)        0.9      0.5
  Bias                       ---     -0.4

The weighted sum is:

$$
z = (0.8)(0.7) + (0.9)(0.5) - 0.4 = 0.61
$$

A step activation would classify the result according to its threshold.
Modern neural networks typically use differentiable activation functions
rather than a hard step function during gradient-based training.

### 2.3 Limitation of a single perceptron

A single perceptron can represent a linear decision boundary. It cannot
correctly model every nonlinear relationship. For example, the XOR
problem is not linearly separable.

------------------------------------------------------------------------

## 3. 🧠 What Is a Multi-Layer Perceptron?

An MLP combines many neurons into layers. Hidden layers transform the
inputs into increasingly useful representations. Nonlinear activation
functions allow the network to model nonlinear relationships.

### Why multiple layers?

A single linear transformation followed by another linear transformation
is still equivalent to one linear transformation. Hidden-layer
**nonlinearity** is therefore essential for learning complex nonlinear
functions.

### MLP characteristics

-   Usually feed-forward: information moves from input toward output.
-   Has one or more hidden layers.
-   Uses trainable weights and biases.
-   Commonly trained with backpropagation and a gradient-based
    optimizer.
-   Can solve nonlinear classification and regression tasks.
-   Does not inherently preserve spatial or sequential structure the way
    specialized CNNs or recurrent/attention-based models can.

------------------------------------------------------------------------

## 4. 🏗️ MLP Architecture

### 4.1 Main layers

  Layer             Purpose                              Trainable parameters?
  ----------------- ------------------------------------ -----------------------
  Input layer       Receives the feature vector          Usually no
  Hidden layer(s)   Learn intermediate representations   Yes
  Output layer      Produces predictions                 Yes

An input layer with 10 features means each example is represented by 10
input values. It does not mean the model has 10 hidden neurons.

### 4.2 Architecture diagram

``` mermaid
flowchart LR
    subgraph Input["Input Layer"]
        I1["x₁"]
        I2["x₂"]
        I3["x₃"]
    end
    subgraph Hidden1["Hidden Layer 1"]
        H11["h₁"]
        H12["h₂"]
        H13["h₃"]
        H14["h₄"]
    end
    subgraph Hidden2["Hidden Layer 2"]
        H21["g₁"]
        H22["g₂"]
        H23["g₃"]
    end
    subgraph Output["Output Layer"]
        O1["ŷ"]
    end
    I1 --> H11
    I1 --> H12
    I1 --> H13
    I1 --> H14
    I2 --> H11
    I2 --> H12
    I2 --> H13
    I2 --> H14
    I3 --> H11
    I3 --> H12
    I3 --> H13
    I3 --> H14
    H11 --> H21
    H12 --> H21
    H13 --> H22
    H14 --> H23
    H21 --> O1
    H22 --> O1
    H23 --> O1
```

This is an illustrative fully connected architecture. In a dense layer,
each neuron connects to every output unit in the next layer.

### 4.3 Neuron computation in a layer

For layer $$l$$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})
$$

The output activation of one layer becomes the input activation of the
next layer.

### 4.4 Parameter count

For a fully connected layer with $$n_{\text{in}}$$ inputs and
$$n_{\text{out}}$$ neurons:

$$
\text{Parameters} = n_{\text{in}}n_{\text{out}} + n_{\text{out}}
$$

The first term counts weights; the second counts one bias per output
neuron.

**Example:** a layer with 8 input features and 5 neurons has:

$$
8 \times 5 + 5 = 45 \text{ parameters}
$$

For a network with layer sizes $$n_0,n_1,\ldots,n_L$$:

$$
P = \sum_{l=1}^{L}(n_{l-1}n_l+n_l)
$$

------------------------------------------------------------------------

## 5. 📖 Core Terminology

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Feature                             Input variable used for prediction

  Label / target                      Correct answer for a training
                                      example

  Neuron / unit                       Computational unit that applies a
                                      weighted sum and activation

  Weight                              Learnable coefficient controlling
                                      an input's influence

  Bias                                Learnable offset

  Logit                               Raw output before a probability
                                      transformation such as softmax

  Activation                          Output of a neuron after applying
                                      its activation function

  Epoch                               One full pass through the training
                                      dataset

  Batch                               Subset of examples used for one
                                      optimization step

  Batch size                          Number of examples in a batch

  Iteration / step                    One optimizer update, usually after
                                      one batch

  Learning rate                       Controls the size of parameter
                                      updates

  Loss                                Objective measuring prediction
                                      error

  Optimizer                           Algorithm that updates trainable
                                      parameters

  Gradient                            Derivative of the loss with respect
                                      to a parameter

  Backpropagation                     Efficient computation of gradients
                                      using the chain rule

  Generalization                      Performance on unseen examples

  Overfitting                         Model learns training-specific
                                      details and performs poorly on new
                                      data

  Underfitting                        Model is too limited or
                                      insufficiently trained to capture
                                      useful patterns
  -----------------------------------------------------------------------

### Epoch, batch, and iteration example

With 1,000 training examples and batch size 100:

$$
\text{Steps per epoch} = \left\lceil \frac{1000}{100} \right\rceil = 10
$$

Ten optimizer steps make one epoch, assuming all examples are used and
the final partial batch is retained.

------------------------------------------------------------------------

## 6. 📐 Mathematical Foundation

### 6.1 Vector notation

For a batch of examples, computations can be vectorized for efficiency.
A dense layer applies matrix multiplication followed by bias addition
and an element-wise activation.

$$
\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]}\left(\mathbf{W}^{[l]}\right)^T + \mathbf{b}^{[l]}
$$

This form assumes examples are stored in rows. Libraries may use
different internal conventions for matrix orientation.

### 6.2 A two-layer MLP

For one hidden layer:

$$
\mathbf{H} = f(\mathbf{X}\mathbf{W}_1+\mathbf{b}_1)
$$

$$
\hat{\mathbf{Y}} = g(\mathbf{H}\mathbf{W}_2+\mathbf{b}_2)
$$

Here, $$f$$ is a hidden activation and $$g$$ is selected according to
the task. For binary classification, the output may use sigmoid; for
multiclass classification, softmax or a logits-based loss is common; for
regression, a linear output is common.

### 6.3 Why nonlinear activations matter

Without nonlinear activations, a stack of dense layers collapses to a
single affine transformation:

$$
\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2
= \mathbf{W}'\mathbf{x}+\mathbf{b}'
$$

Therefore, depth alone does not create nonlinear modeling capacity;
nonlinear operations are crucial.

------------------------------------------------------------------------

## 7. ⚡ Activation Functions

Activation functions introduce nonlinearity or control the output range.

### 7.1 Common functions

  ------------------------------------------------------------------------------------------
  Function          Formula                              Typical use       Notes
  ----------------- ------------------------------------ ----------------- -----------------
  Sigmoid           $$\sigma(z)=\frac{1}{1+e^{-z}}$$     Binary output     Saturates at
                                                         probability       extreme values;
                                                                           gradients can
                                                                           become small

  Tanh              $$\tanh(z)$$                         Some hidden       Output range is
                                                         layers            -1 to 1; can
                                                                           saturate

  ReLU              $$\max(0,z)$$                        Common            Simple and
                                                         hidden-layer      efficient; can
                                                         default           have inactive
                                                                           units

  Leaky ReLU        $$\max(\alpha z,z)$$                 Hidden layers     Keeps a small
                                                                           negative-side
                                                                           slope

  GELU              $$z\Phi(z)$$                         Modern neural     Smooth
                                                         architectures     activation; more
                                                                           computationally
                                                                           involved

  Softmax           $$\frac{e^{z_i}}{\sum_j e^{z_j}}$$   Multiclass output Produces values
                                                                           summing to 1
  ------------------------------------------------------------------------------------------

### 7.2 ReLU

$$
f(z)=\max(0,z)
$$

ReLU outputs zero for negative inputs and the input itself for positive
inputs. It is often a strong default for hidden layers.

### 7.3 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Sigmoid maps a logit to a value between 0 and 1. With a suitable loss
and threshold, this can represent a binary-class probability estimate.

### 7.4 Softmax

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Softmax converts a vector of logits into normalized class probabilities.
For numerical stability, implementations typically subtract the maximum
logit before exponentiation.

### 7.5 Choosing an output activation

  -----------------------------------------------------------------------
  Problem                 Output design           Common loss
  ----------------------- ----------------------- -----------------------
  Regression              Linear output           Mean squared error or
                                                  mean absolute error

  Binary classification   One logit or sigmoid    Binary cross-entropy /
                          probability             logits-based binary
                                                  cross-entropy

  Single-label multiclass One logit per class;    Categorical
  classification          softmax probabilities   cross-entropy or sparse
                          when needed             categorical
                                                  cross-entropy

  Multi-label             One independent logit   Binary cross-entropy
  classification          per label               per label
  -----------------------------------------------------------------------

**Important:** In frameworks such as Keras, a logits-based loss should
be configured with `from_logits=True` when the model outputs raw logits.
Do not apply softmax twice.

------------------------------------------------------------------------

## 8. ➡️ Forward Propagation

Forward propagation calculates the model's prediction from its input.

### Process

1.  Supply the input features.
2.  Multiply each input by its weights.
3.  Add biases.
4.  Apply the layer activation.
5.  Pass activations through each subsequent layer.
6.  Produce the final output.

``` mermaid
flowchart TD
    A[Input features X] --> B[Dense transformation: W₁X + b₁]
    B --> C[Hidden activation]
    C --> D[Dense transformation: W₂H + b₂]
    D --> E[Output transformation]
    E --> F[Prediction ŷ]
```

### Example

Consider an MLP with 3 input features, 4 hidden neurons, and 1 output
neuron. The hidden layer produces four activations; the output layer
combines them to produce one prediction. During training, the prediction
is compared with the target using a loss function.

Forward propagation alone does not improve the model. Learning occurs
when gradients from the loss are used to update parameters.

------------------------------------------------------------------------

## 9. 📉 Loss Functions

A loss function quantifies how far predictions are from targets.
Training attempts to minimize the chosen objective.

### 9.1 Mean Squared Error (MSE)

For regression:

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

MSE penalizes large errors strongly because the errors are squared.

### 9.2 Mean Absolute Error (MAE)

$$
MAE=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

MAE is less sensitive to large individual errors than MSE.

### 9.3 Binary cross-entropy

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

Here, $$y_i$$ is 0 or 1 and $$p_i$$ is the predicted probability of
class 1. Practical implementations clip probabilities or operate
directly on logits for numerical stability.

### 9.4 Multiclass cross-entropy

For one-hot targets:

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(p_{ik})
$$

If labels are integer class indices, sparse categorical cross-entropy
avoids the need to manually one-hot encode targets.

### 9.5 Loss vs metric

  -----------------------------------------------------------------------
  Loss                                Metric
  ----------------------------------- -----------------------------------
  Optimized directly during training  Used to assess performance

  Must support useful gradients for   May be non-differentiable
  the training method                 

  Examples: cross-entropy, MSE        Examples: accuracy, F1-score, AUC
  -----------------------------------------------------------------------

A model can have decreasing loss while a threshold-based metric remains
unchanged. Always evaluate metrics relevant to the problem.

------------------------------------------------------------------------

## 10. 🔁 Backpropagation and Gradient Descent

### 10.1 What is backpropagation?

Backpropagation computes how the loss changes with respect to each
trainable weight and bias. It applies the chain rule from the output
layer back through earlier layers.

It calculates gradients; the optimizer uses those gradients to update
parameters.

### 10.2 Gradient descent update

$$
w_{\text{new}}=w_{\text{old}}-\eta\frac{\partial L}{\partial w}
$$

$$
b_{\text{new}}=b_{\text{old}}-\eta\frac{\partial L}{\partial b}
$$

Where $$\eta$$ is the learning rate and $$L$$ is the loss.

### 10.3 Backpropagation workflow

``` mermaid
flowchart TD
    A[Input batch] --> B[Forward pass]
    B --> C[Compute loss]
    C --> D[Compute output-layer gradients]
    D --> E[Propagate gradients backward]
    E --> F[Compute gradients for all parameters]
    F --> G[Optimizer updates weights and biases]
    G --> H{More batches or epochs?}
    H -- Yes --> A
    H -- No --> I[Evaluate model]
```

### 10.4 Chain rule intuition

If the loss depends on an output, and the output depends on a hidden
activation, and the activation depends on a weight, the derivative links
these dependencies together.

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial h}
\frac{\partial h}{\partial w}
$$

This is a simplified chain-rule illustration for one path through the
network. Actual networks sum gradient contributions from all relevant
paths.

### 10.5 Common optimizers

  -----------------------------------------------------------------------
  Optimizer               Main idea               Notes
  ----------------------- ----------------------- -----------------------
  SGD                     Uses gradients from a   Simple; may need
                          batch                   careful learning-rate
                                                  tuning

  Momentum SGD            Accumulates a           Can smooth noisy
                          velocity-like moving    updates
                          direction               

  RMSprop                 Adapts updates using a  Useful in some
                          moving average of       nonstationary settings
                          squared gradients       

  Adam                    Uses moving averages of Common starting choice
                          gradients and squared   
                          gradients               

  AdamW                   Decouples weight decay  Often preferred when
                          from Adam's gradient    weight decay is desired
                          update                  
  -----------------------------------------------------------------------

An optimizer is not a substitute for correct preprocessing, a suitable
loss, or sound validation.

------------------------------------------------------------------------

## 11. 🧪 Training Workflow

### 11.1 Prepare the data

-   Inspect missing values, duplicates, data types, and target
    distribution.
-   Encode categorical variables appropriately.
-   Scale numerical features, especially for MLPs.
-   Split into training, validation, and test sets.
-   Fit preprocessing only on the training set to avoid data leakage.

### 11.2 Train, validate, and test

  -----------------------------------------------------------------------
  Dataset split                       Purpose
  ----------------------------------- -----------------------------------
  Training                            Fit weights and biases

  Validation                          Select architecture,
                                      hyperparameters, and stopping point

  Test                                Final unbiased evaluation after
                                      choices are complete
  -----------------------------------------------------------------------

Do not repeatedly tune against the test set. That leaks information from
the test set into model selection.

### 11.3 Standardization

A common transformation is:

$$
z=\frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Calculate $$\mu_{\text{train}}$$ and $$\sigma_{\text{train}}$$ from
training data only, then use those same values to transform validation,
test, and future data.

### 11.4 Learning curves

-   Training and validation loss both high: possible underfitting,
    inadequate features, or optimization problems.
-   Training loss falls while validation loss rises: likely overfitting.
-   Both losses fluctuate sharply: learning rate may be too high or data
    may be unstable.
-   Both improve slowly: learning rate may be too low, model capacity
    may be limited, or features may be poorly scaled.

------------------------------------------------------------------------

## 12. 🐍 Implementing an MLP with Scikit-learn

Scikit-learn provides `MLPClassifier` and `MLPRegressor` for tabular
classification and regression.

### 12.1 Classification example

This example uses the built-in breast cancer dataset and a pipeline so
scaling is learned only from the training split.

``` python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Hold out the test set before model fitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling is fitted on training data inside the pipeline
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
```

### Explanation

-   `hidden_layer_sizes=(64, 32)` creates two hidden layers with 64 and
    32 units.
-   `activation="relu"` uses ReLU in hidden layers.
-   `alpha` controls L2 regularization strength.
-   `learning_rate_init` sets the initial learning rate for the Adam
    solver.
-   `early_stopping=True` reserves a portion of the training data for
    internal validation and stops when validation performance no longer
    improves according to the estimator's rule.
-   `stratify=y` preserves class proportions in the split.

The dataset's class labels and metrics should be interpreted in the
context of the dataset. This demonstration is educational, not a
clinical diagnostic system.

### 12.2 Predicting probabilities

``` python
probabilities = model.predict_proba(X_test)
print(probabilities[:5])
```

Each row contains estimated probabilities for the classes. Probability
calibration is not guaranteed merely because a model returns
probabilities.

### 12.3 Regression with `MLPRegressor`

``` python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

regressor = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))
```

The first run may download the California housing dataset. Check the
dataset's license, suitability, and current availability before using it
in a production workflow.

------------------------------------------------------------------------

## 13. 🧰 Implementing an MLP with TensorFlow/Keras

Keras is useful when you need more control over training loops,
callbacks, GPU acceleration, or model deployment.

### 13.1 Binary classification template

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# X_train and y_train should be prepared NumPy arrays or compatible tensors.
# Fit any preprocessing on training data only.

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

test_results = model.evaluate(X_test, y_test, return_dict=True)
print(test_results)
```

This is a template, not a standalone script: define `X_train`,
`y_train`, `X_val`, `y_val`, `X_test`, and `y_test` after making
leakage-safe splits and applying suitable preprocessing.

### 13.2 Multiclass classification

For integer class labels from 0 through `K - 1`, use `Dense(K)` with a
logits-based loss:

``` python
num_classes = 4

model = keras.Sequential([
    layers.Input(shape=(number_of_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes)  # Raw logits
])

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

To convert logits to probabilities at inference time:

``` python
logits = model.predict(X_new)
probabilities = tf.nn.softmax(logits, axis=1).numpy()
predicted_classes = probabilities.argmax(axis=1)
```

### 13.3 Saving and loading a model

``` python
model.save("mlp_model.keras")

loaded_model = keras.models.load_model("mlp_model.keras")
```

Save preprocessing artifacts alongside the model. A trained network
without the matching feature order, scaler, encoders, and label mapping
may make incorrect predictions.

------------------------------------------------------------------------

## 14. 🧩 Practical Mini-Project: Classification

### Project: Predict whether a tumor record is benign or malignant

**Goal:** Train an MLP on a public, built-in educational dataset and
evaluate its classification performance.

### Project workflow

``` mermaid
flowchart TD
    A[Load dataset] --> B[Explore features and labels]
    B --> C[Stratified train/test split]
    C --> D[Build scaling + MLP pipeline]
    D --> E[Train model]
    E --> F[Evaluate accuracy, precision, recall, F1]
    F --> G[Inspect confusion matrix]
    G --> H[Document limitations and results]
```

### Suggested project structure

``` text
mlp-classification-project/
├── train_mlp.py
├── requirements.txt
├── README.md
└── reports/
    └── evaluation.md
```

### Suggested dependencies

``` text
scikit-learn
numpy
```

### Evaluation checklist

-   [ ] Use a stratified train/test split.
-   [ ] Fit the scaler on training data only.
-   [ ] Record random seed and hyperparameters.
-   [ ] Report precision, recall, F1-score, and confusion matrix, not
    just accuracy.
-   [ ] Inspect class imbalance and the cost of false positives/false
    negatives.
-   [ ] Compare with a simple baseline such as logistic regression.
-   [ ] Keep the test set out of model selection.
-   [ ] State that a classroom dataset model is not a validated medical
    device.

### Example of interpreting a confusion matrix

For binary classification, rows often represent true labels and columns
represent predicted labels (confirm the library's convention).

                       Predicted positive    Predicted negative
  ----------------- --------------------- ---------------------
  Actual positive      True positive (TP)   False negative (FN)
  Actual negative     False positive (FP)    True negative (TN)

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

Precision measures the reliability of positive predictions. Recall
measures the proportion of actual positives detected. Select the
priority metric based on the real-world cost of errors.

------------------------------------------------------------------------

## 15. 🌍 Use Cases and Real-World Examples

  -----------------------------------------------------------------------
  Domain            Example task      Inputs            Output
  ----------------- ----------------- ----------------- -----------------
  Finance           Credit-risk       Income, repayment Risk score or
                    scoring support   history, debt     class
                                      ratios            

  Retail            Customer churn    Usage, purchase   Churn probability
                    prediction        history, support  
                                      interactions      

  Healthcare        Classification of Numeric clinical  Predicted
  research          structured        features          category
                    measurements                        

  Manufacturing     Predictive        Sensor summaries, Failure-risk
                    maintenance       operating         estimate
                                      conditions        

  Marketing         Conversion        Campaign,         Conversion
                    prediction        engagement, user  probability
                                      features          

  Energy            Demand estimation Weather,          Numeric forecast
                                      calendar,         
                                      historical values 

  Education         Student           Attendance,       Predicted outcome
                    performance       assessment        
                    analysis          history           

  Computer vision   Small             Pixel values or   Image class
                    flattened-image   learned features  
                    classification                      
                    tasks                               
  -----------------------------------------------------------------------

MLPs can be useful for structured data, but tree-based methods often
perform very competitively on tabular datasets. For images, raw
flattened pixels may discard spatial structure; CNNs or vision
transformers are often more suitable.

------------------------------------------------------------------------

## 16. ⚖️ Advantages and Limitations

### Advantages

-   Learns nonlinear relationships.
-   Supports classification and regression.
-   Flexible architecture and output design.
-   Can model interactions between input features.
-   Works with many numeric feature representations.
-   Has broad library support and can be accelerated with suitable
    hardware.

### Limitations

-   Often requires feature scaling.
-   Can be sensitive to hyperparameters and random initialization.
-   May need substantial data for complex tasks.
-   Less interpretable than many simpler models.
-   Can overfit, especially with excessive capacity or limited data.
-   Training can be less predictable than tree-based methods on small
    tabular datasets.
-   Does not automatically handle missing or categorical data without
    appropriate preprocessing.
-   A good training score does not establish fairness, calibration, or
    real-world reliability.

### When should you choose an MLP?

Consider an MLP when you have enough examples, meaningful numeric
representations, nonlinear patterns, and a validation setup for
comparing it with simpler baselines. Do not assume it is the best choice
solely because it is a neural network.

------------------------------------------------------------------------

## 17. 🔍 MLP vs Other Models

  ---------------------------------------------------------------------------------------
  Model             Strengths         Limitations             Often useful for
  ----------------- ----------------- ----------------------- ---------------------------
  Linear/logistic   Fast,             Limited to linear       Small datasets, baseline
  regression        interpretable     relationships unless    modeling
                    baseline          features are engineered 

  Decision tree     Easy to explain;  Can overfit without     Rule-like patterns
                    captures          controls                
                    nonlinear rules                           

  Random forest     Robust, strong    Larger models; less     General tabular
                    tabular baseline  compact interpretation  classification/regression

  Gradient boosting Often excellent   Tuning and training can Structured datasets
                    on tabular data   be more involved        

  MLP               Flexible          Scaling and tuning      Numeric data, learned
                    nonlinear         sensitivity; less       representations
                    function          transparent             
                    approximator                              

  CNN               Uses local        Architecture-specific   Images and spatial grids
                    spatial structure and data/hardware needs 

  Transformer       Models            Can be computationally  Text, sequences, multimodal
                    relationships     expensive               tasks
                    through attention                         
  ---------------------------------------------------------------------------------------

No model is universally best. Evaluate models using the same data
splits, metrics, and leakage-safe preprocessing.

------------------------------------------------------------------------

## 18. 🎛️ Hyperparameter Tuning

Hyperparameters are set before or around training; they are not learned
as ordinary model weights.

  -----------------------------------------------------------------------
  Hyperparameter          What it controls        Practical starting
                                                  approach
  ----------------------- ----------------------- -----------------------
  Hidden layers           Depth and               Start with one or two
                          representational        
                          capacity                

  Units per layer         Width/capacity          Try modest widths, then
                                                  compare

  Learning rate           Update size             Start near 0.001 for
                                                  Adam, then tune

  Batch size              Examples per update     Try 32, 64, or 128
                                                  depending on data

  Epochs                  Maximum training passes Use validation and
                                                  early stopping

  Activation              Nonlinear               ReLU is a common
                          transformation          hidden-layer baseline

  Weight decay / L2       Penalizes large weights Tune using validation
                                                  performance

  Dropout                 Randomly disables       Try modest rates if
                          activations during      overfitting
                          training                

  Optimizer               Parameter update rule   Adam/AdamW or SGD with
                                                  momentum

  Random seed             Reproducibility control Record it; compare
                                                  multiple seeds when
                                                  feasible
  -----------------------------------------------------------------------

### A systematic tuning strategy

1.  Establish a simple baseline.
2.  Confirm the data split and preprocessing.
3.  Tune learning rate and architecture before adding many
    regularization techniques.
4.  Use validation metrics and early stopping.
5.  Compare a small, planned search rather than changing many settings
    at once.
6.  Re-run promising settings with multiple seeds when results vary.
7.  Evaluate the final selected model once on the held-out test set.

------------------------------------------------------------------------

## 19. 🛡️ Regularization and Generalization

Regularization reduces the risk that the network memorizes
training-specific noise.

### 19.1 L2 regularization

A common objective adds a penalty on weights:

$$
L_{\text{total}}=L_{\text{data}}+\lambda\sum_j w_j^2
$$

Here, $$\lambda$$ controls the penalty strength. Biases are often
excluded from this penalty, depending on the implementation.

### 19.2 Dropout

Dropout randomly sets a fraction of activations to zero during training.
It is disabled during ordinary inference, with scaling handled by the
framework.

### 19.3 Early stopping

Stop training when a monitored validation metric stops improving for a
configured patience period. Restoring the best weights can prevent
keeping a later, overfit state.

### 19.4 Other useful methods

-   More representative training data.
-   Feature selection or domain-informed feature engineering.
-   Appropriate model capacity.
-   Learning-rate schedules.
-   Weight decay.
-   Data augmentation where valid for the data type.
-   Cross-validation when data is limited, while keeping a final test
    set separate.

### Underfitting vs overfitting

  -----------------------------------------------------------------------
  Symptom                 Likely issue            Possible response
  ----------------------- ----------------------- -----------------------
  High training and       Underfitting or         Increase capacity,
  validation error        optimization failure    train longer, improve
                                                  features, check
                                                  learning rate

  Low training error,     Overfitting             Add regularization,
  high validation error                           simplify model, gather
                                                  data

  Validation score varies Training instability or Tune optimization, use
  greatly by seed         small validation set    multiple seeds, improve
                                                  evaluation design

  Excellent validation    Distribution shift,     Audit splits and
  but poor test result    leakage, or over-tuning pipeline; assess data
                                                  drift
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 20. 🚀 Advanced Concepts

### 20.1 Universal approximation

Under suitable conditions, feed-forward networks with at least one
sufficiently wide hidden layer and a suitable nonlinear activation can
approximate broad classes of continuous functions on compact domains.
This is an existence result, not a guarantee that training will find a
good solution or that the required network will be small.

### 20.2 Vanishing and exploding gradients

Gradients may shrink or grow as they pass through many layers.
Saturating activations such as sigmoid or tanh can contribute to
vanishing gradients. Poor initialization or unstable optimization can
also cause gradient problems.

Potential mitigations include:

-   ReLU-family activations in suitable hidden layers.
-   Xavier/Glorot or He initialization, as appropriate.
-   Feature normalization and sensible input scaling.
-   Carefully chosen learning rates.
-   Residual connections in deeper architectures.
-   Gradient clipping in training setups where exploding gradients
    occur.

### 20.3 Weight initialization

Weights should not all start at the same value, because symmetry can
prevent hidden units from learning distinct features. Modern libraries
generally use sensible defaults, but initialization remains important
for deeper networks.

-   **Glorot/Xavier initialization:** commonly used with tanh-like
    activations.
-   **He/Kaiming initialization:** designed for ReLU-family activations.

### 20.4 Batch normalization

Batch normalization normalizes intermediate activations using batch
statistics during training and moving statistics during inference. It
can improve optimization in some networks, but is not always necessary
for a small MLP.

### 20.5 Learning-rate schedules

The learning rate can be reduced during training when progress slows.
Examples include step decay, cosine schedules, and reduce-on-plateau
approaches.

### 20.6 Mini-batch optimization

-   **Batch gradient descent:** uses the full training set for each
    update.
-   **Stochastic gradient descent:** uses one example per update.
-   **Mini-batch gradient descent:** uses a subset of examples per
    update.

Mini-batches are widely used because they balance computational
efficiency and noisy gradient estimates.

### 20.7 Class imbalance and threshold selection

Accuracy can be misleading when one class dominates. Use class-specific
precision/recall, macro F1, PR-AUC or ROC-AUC as appropriate, and
consider class weighting or resampling only within the training process.
For binary classification, a threshold of 0.5 is not automatically
optimal; choose it using validation data and the real cost of errors.

### 20.8 Calibration and explainability

A predicted probability is not necessarily a well-calibrated
probability. Evaluate calibration when probabilities drive decisions.
For explanations, consider feature permutation importance, SHAP-style
methods, or carefully designed sensitivity analysis, noting that
explanations can be unstable and do not establish causality.

### 20.9 Reproducibility

Record:

-   Dataset version and split method.
-   Preprocessing steps and fitted artifacts.
-   Model architecture and parameter count.
-   Library versions and hardware.
-   Random seeds.
-   Hyperparameters and stopping criteria.
-   Evaluation metrics and the exact test protocol.

Perfect reproducibility may not be guaranteed across hardware, software
versions, or nondeterministic operations.

------------------------------------------------------------------------

## 21. ✅ Best Practices

1.  **Start with a baseline.** Compare the MLP with logistic regression
    and a tree-based model.
2.  **Scale numeric features.** MLP optimization is often more stable
    when input scales are comparable.
3.  **Prevent leakage.** Fit scalers, imputers, encoders, and feature
    selection on training data only.
4.  **Use the right output/loss pair.** Ensure target shape and encoding
    match the loss.
5.  **Monitor validation metrics.** Do not use test results to make
    repeated tuning decisions.
6.  **Use early stopping where appropriate.** It can reduce unnecessary
    training and overfitting.
7.  **Track class-wise performance.** Overall accuracy can hide poor
    minority-class recall.
8.  **Keep a reproducible pipeline.** Save preprocessing and model
    together.
9.  **Inspect failure cases.** Look for systematic errors by class,
    subgroup, or operating condition.
10. **Avoid unnecessary complexity.** A larger network is not
    automatically a better model.
11. **Check the deployment distribution.** Monitor drift and performance
    after deployment.
12. **Treat high-stakes predictions carefully.** Require domain review,
    appropriate validation, privacy protection, and relevant regulatory
    checks.

------------------------------------------------------------------------

## 22. 🧯 Common Mistakes and Troubleshooting

  -----------------------------------------------------------------------
  Mistake                 Why it is a problem     Fix
  ----------------------- ----------------------- -----------------------
  Not scaling features    Optimization may be     Use a training-fitted
                          slow or unstable        scaler

  Applying preprocessing  Leaks test-set          Split first; fit
  before splitting        information             transformations on
                                                  training data

  Using sigmoid for       Output setup may not    Use one logit per class
  mutually exclusive      match the task          with a suitable
  multi-class output                              multiclass loss

  Applying softmax and    Loss receives           Use either logits with
  also setting            probabilities while     `from_logits=True`, or
  `from_logits=True`      expecting logits        probabilities with the
                                                  matching loss

  Using accuracy alone    Hides class-specific    Report precision,
                          failure                 recall, F1, and
                                                  confusion matrix

  Setting learning rate   Loss may diverge or     Lower the learning rate
  too high                oscillate               

  Using too many neurons  Raises overfitting risk Reduce capacity or
  on little data                                  regularize

  Training for too few    Model may not converge  Inspect learning curves
  epochs                                          and validation behavior

  Training for too many   May overfit             Use early stopping
  epochs                                          

  Forgetting feature      Inputs may be           Persist and validate
  order at inference      interpreted incorrectly the exact preprocessing
                                                  pipeline

  Repeatedly tuning on    Test score becomes      Keep a final untouched
  the test set            biased                  test set

  Comparing models on     Comparison is           Use consistent splits
  different splits        unreliable              and metrics
  -----------------------------------------------------------------------

### Debugging checklist

-   Are features numeric and finite?
-   Are labels encoded as expected?
-   Does the final layer have the right number of units?
-   Does the loss match the target encoding?
-   Are gradients and loss finite?
-   Is the learning rate reasonable?
-   Does training loss decrease?
-   Is validation performance monitored independently?
-   Is the class distribution checked?
-   Are evaluation metrics appropriate to the task?

------------------------------------------------------------------------

## 23. 🎤 Interview Questions and Answers

### Q1. What is an MLP?

An MLP is a feed-forward neural network with an input layer, one or more
hidden layers, and an output layer. It learns weights and biases using a
training objective and an optimization algorithm.

### Q2. What is the difference between a perceptron and an MLP?

A single perceptron represents a linear decision boundary in its input
space. An MLP with nonlinear hidden activations can learn nonlinear
decision boundaries.

### Q3. Why are activation functions required?

Without nonlinear activations, stacked dense layers reduce to a single
affine transformation. Nonlinear activations allow the network to
represent more complex functions.

### Q4. What is forward propagation?

Forward propagation passes input features through the network to compute
predictions.

### Q5. What is backpropagation?

Backpropagation uses the chain rule to compute derivatives of the loss
with respect to the network's trainable parameters.

### Q6. Are backpropagation and gradient descent the same?

No. Backpropagation computes gradients; gradient descent or another
optimizer uses those gradients to update parameters.

### Q7. What is the role of a bias?

A bias shifts a neuron's pre-activation value and allows a neuron to fit
patterns that do not pass through the origin.

### Q8. What is an epoch?

An epoch is one complete pass through the training dataset.

### Q9. What is the learning rate?

The learning rate controls the scale of parameter updates. A value that
is too high can destabilize training; one that is too low can make
training slow.

### Q10. Why scale data for an MLP?

Feature scaling can improve optimization by preventing large-magnitude
features from dominating gradient behavior.

### Q11. How do you prevent overfitting?

Use validation-based early stopping, regularization, suitable model
capacity, representative data, and careful evaluation design.

### Q12. Which activation is commonly used in hidden layers?

ReLU is a common default, though the best choice depends on the
architecture and task.

### Q13. Which loss should be used for regression?

MSE and MAE are common choices. The appropriate choice depends on the
error distribution and business objective.

### Q14. Which loss is used for multi-class classification?

Categorical cross-entropy is common for one-hot targets; sparse
categorical cross-entropy is common for integer class labels.

### Q15. Why might an MLP perform worse than a random forest on tabular data?

Tree ensembles often handle heterogeneous feature scales, nonlinear
thresholds, and interactions effectively with less preprocessing. MLPs
can still work well, but need suitable representation, scaling,
architecture, and tuning.

### Q16. What is the universal approximation theorem?

It describes conditions under which a neural network can approximate a
broad class of functions. It does not guarantee easy optimization, good
generalization, or efficient approximation with limited resources.

### Q17. What is the difference between a parameter and a hyperparameter?

Weights and biases are learned parameters. Learning rate, hidden-layer
sizes, batch size, and regularization strength are examples of
hyperparameters.

### Q18. What is the difference between validation and test data?

Validation data guides model selection and tuning. Test data is reserved
for final evaluation after those choices are complete.

### Q19. What is a logit?

A logit is a raw model score before applying a probability
transformation such as sigmoid or softmax.

### Q20. What should you inspect besides accuracy?

Consider precision, recall, F1-score, confusion matrix, calibration,
class imbalance, error costs, and performance on relevant subgroups.

------------------------------------------------------------------------

## 24. ⚡ Quick Revision

### 24.1 Key points

-   An MLP is a feed-forward neural network with one or more hidden
    layers.
-   Weights and biases are the learnable parameters.
-   Forward propagation produces predictions.
-   A loss function measures prediction error.
-   Backpropagation calculates gradients.
-   The optimizer updates parameters using those gradients.
-   Nonlinear activation functions allow nonlinear modeling.
-   Feature scaling is usually important for numeric MLP inputs.
-   Use validation data for model selection and reserve test data for
    final evaluation.
-   Monitor more than accuracy, especially with imbalanced classes.
-   Compare against simpler and tree-based baselines.

### 24.2 Important formulas

**Neuron**

$$
z=\mathbf{w}^{T}\mathbf{x}+b,\qquad a=f(z)
$$

**Layer**

$$
\mathbf{a}^{[l]}=f^{[l]}(\mathbf{W}^{[l]}\mathbf{a}^{[l-1]}+\mathbf{b}^{[l]})
$$

**ReLU**

$$
f(z)=\max(0,z)
$$

**Sigmoid**

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

**Softmax**

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**Gradient update**

$$
\theta_{\text{new}}=\theta_{\text{old}}-\eta\nabla_{\theta}L
$$

**MSE**

$$
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

**Number of parameters in one dense layer**

$$
P=n_{\text{in}}n_{\text{out}}+n_{\text{out}}
$$

### 24.3 Important Python classes and methods

  -----------------------------------------------------------------------
  Library                 Class / method          Purpose
  ----------------------- ----------------------- -----------------------
  Scikit-learn            `MLPClassifier`         MLP classification

  Scikit-learn            `MLPRegressor`          MLP regression

  Scikit-learn            `StandardScaler`        Standardize numeric
                                                  features

  Scikit-learn            `Pipeline` /            Keep preprocessing and
                          `make_pipeline`         estimator together

  Keras                   `layers.Dense`          Fully connected layer

  Keras                   `model.compile()`       Configure optimizer,
                                                  loss, and metrics

  Keras                   `model.fit()`           Train model

  Keras                   `model.evaluate()`      Evaluate model

  Keras                   `model.predict()`       Generate outputs

  Keras                   `EarlyStopping`         Stop when monitored
                                                  validation performance
                                                  stalls
  -----------------------------------------------------------------------

### 24.4 Visual summary / roadmap

``` mermaid
flowchart TD
    A["Inputs: features X"] --> B["Dense layer: weights + bias"]
    B --> C["Activation: nonlinear transformation"]
    C --> D["Repeat hidden layers"]
    D --> E["Output layer"]
    E --> F["Prediction"]
    F --> G["Loss versus target"]
    G --> H["Backpropagation: compute gradients"]
    H --> I["Optimizer: update parameters"]
    I --> J{"Stop condition met?"}
    J -- No --> B
    J -- Yes --> K["Validate and test"]
    K --> L["Deploy with preprocessing and monitoring"]
```

### 24.5 One-minute recap

  -----------------------------------------------------------------------
  Question                            Answer
  ----------------------------------- -----------------------------------
  What does MLP stand for?            Multi-Layer Perceptron

  What makes it nonlinear?            Nonlinear activation functions

  What creates predictions?           Forward propagation

  What calculates gradients?          Backpropagation

  What changes weights?               Optimizer

  What measures error?                Loss function

  What helps control overfitting?     Regularization, early stopping,
                                      data, and model selection

  Why scale features?                 To improve optimization behavior

  What selects hyperparameters?       Validation data

  What gives final performance        A held-out test set used after
  estimate?                           model selection
  -----------------------------------------------------------------------

### 24.6 Learning roadmap

1.  Understand a single neuron and the XOR limitation.
2.  Learn dense layers, weights, biases, and activation functions.
3.  Work through forward propagation numerically.
4.  Learn loss functions and the chain rule.
5.  Implement an MLP with Scikit-learn.
6.  Train a Keras model and inspect learning curves.
7.  Practice scaling, stratified splits, and leakage prevention.
8.  Tune architecture and regularization using validation data.
9.  Compare MLP performance against linear models and tree ensembles.
10. Package preprocessing and the model together; document limitations
    and monitor deployment.

------------------------------------------------------------------------

## 📌 Final Takeaway

A Multi-Layer Perceptron learns by repeatedly making predictions,
measuring errors, computing gradients through backpropagation, and
updating weights and biases with an optimizer. Its flexibility makes it
useful for nonlinear classification and regression, but success depends
on data quality, correct preprocessing, suitable architecture, stable
optimization, careful validation, and honest evaluation on unseen data.
