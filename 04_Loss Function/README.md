# 📘 Loss Functions in Deep Learning

## 📑 Table of Contents

1. [🎯 Introduction](#-1-introduction)
2. [🧠 What Is a Loss Function?](#-2-what-is-a-loss-function)
3. [⚙️ How Loss Functions Work](#-3-how-loss-functions-work)
4. [🔤 Important Terminology](#-4-important-terminology)
5. [🧮 Mathematical Foundations](#-5-mathematical-foundations)
6. [📚 Types of Loss Functions](#-6-types-of-loss-functions)
7. [📊 Loss Functions by Task](#-7-loss-functions-by-task)
8. [🆚 Comparison of Common Loss
Functions](#-8-comparison-of-common-loss-functions)
9. [💻 Practical Python Examples](#-9-practical-python-examples)
10. [🧪 Mini-Project: Compare Regression
Losses](#-10-mini-project-compare-regression-losses)
11. [🌍 Real-World Use Cases](#-11-real-world-use-cases)
12. [🚀 Advanced Concepts](#-12-advanced-concepts)
13. [✅ Best Practices](#-13-best-practices)
14. [⚠️ Common Mistakes](#-14-common-mistakes)
15. [🎤 Interview Questions and
Points](#-15-interview-questions-and-points)
16. [⚡ Quick Revision](#-16-quick-revision)

---

# 🎯 1. Introduction

A **loss function** is a mathematical function that measures how far a model's prediction is from the correct target. During training, a deep learning model uses the loss to determine how its parameters should be adjusted.

A model typically learns by repeatedly:

1. Making predictions during the forward pass.
2. Calculating the loss by comparing predictions with target values.
3. Computing gradients through backpropagation.
4. Updating weights and biases using an optimizer.
5. Repeating this process over many batches and epochs.

The loss function is a central part of the learning objective. A poorly chosen loss can encourage the model to optimize the wrong behavior, even if the training process runs correctly.

## 🎯 Learning Objectives

By the end of these notes, you should be able to:

- Explain what a loss function does and how it differs from an
evaluation metric.
- Select appropriate losses for regression, classification,
segmentation, and other tasks.
- Understand the key formulas and assumptions behind common loss
functions.
- Implement losses using NumPy and TensorFlow/Keras or PyTorch.
- Recognize issues such as class imbalance, outliers, saturation, and
incorrect output activations.
- Explain advanced ideas such as weighted losses, focal loss,
contrastive loss, and multi-task loss.

---

# 🧠 2. What Is a Loss Function?

A loss function maps a prediction and its target to a numerical measure of error.

For a single example:

$$
L(y, \hat{y})
$$

Where:

- $$y$$ is the true target.
- $$\hat{y}$$ is the model prediction.
- $$L$$ is the loss value.

For a dataset containing $$N$$ examples, the average empirical loss is often:

$$
J(\theta) = \frac{1}{N}\sum_{i=1}^{N} L\left(y_i, f_\theta(x_i)\right)
$$

Where:

- $$x_i$$ is the input for example $$i$$.
- $$y_i$$ is its target.
- $$f_\theta$$ is the model with parameters $$\theta$$.
- $$J(\theta)$$ is the objective being minimized.

The training objective may also include a regularization term:

$$
J(\theta) = \frac{1}{N}\sum_{i=1}^{N} L\left(y_i, f_\theta(x_i)\right) + \lambda R(\theta)
$$

Here, $$R(\theta)$$ penalizes undesirable parameter configurations, and $$\lambda$$ controls the strength of regularization.

## 💡 Intuition

Suppose a model predicts house prices:

Actual price   Predicted price   Absolute error -------------- ----------------- ---------------- ₹50 lakh          ₹48 lakh          ₹2 lakh ₹70 lakh          ₹75 lakh          ₹5 lakh ₹40 lakh          ₹39 lakh          ₹1 lakh

A loss function converts these differences into a scalar value that can be optimized. Different loss functions penalize the same errors differently.

## 🧩 Loss Function vs. Cost Function vs. Metric

Terminology varies between textbooks and libraries, but a common distinction is:

----------------------------------------------------------------------- Term                    Meaning                 Example ----------------------- ----------------------- ----------------------- Loss                    Error for one example   Cross-entropy for one or a batch              batch

Cost/objective          Aggregate training      Mean batch loss or loss, possibly plus     full-dataset objective regularization

Evaluation metric       Quantity used to assess Accuracy, F1-score, MAE model performance

Optimizer               Algorithm that updates  SGD, Adam parameters using gradients -----------------------------------------------------------------------

A metric is not necessarily differentiable. Accuracy, for example, is useful for evaluation but generally unsuitable as the direct training objective for standard gradient-based learning.

---

# ⚙️ 3. How Loss Functions Work

## 🔄 Training Workflow

``` mermaid
flowchart TD
    A[Input data] --> B[Neural network]
    B --> C[Forward pass and predictions]
    C --> D[Loss function compares predictions with targets]
    D --> E[Scalar loss]
    E --> F[Backpropagation computes gradients]
    F --> G[Optimizer updates parameters]
    G --> H{Training complete?}
    H -- No --> A
    H -- Yes --> I[Evaluate on validation and test data]
```

## 🪜 Step-by-Step Example

Assume the actual regression target is $$y=10$$ and the model predicts $$\hat{y}=7$$.

Using squared error for this example:

$$
L(y,\hat{y}) = (y-\hat{y})^2
$$

$$
L(10,7) = (10-7)^2 = 9
$$

The loss indicates the prediction is not exact. During training, backpropagation calculates how the loss changes with model parameters. The optimizer then adjusts parameters in a direction intended to reduce the objective.

For mean squared error over a batch of $$N$$ examples:

$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

## 📉 Why Gradients Matter

For a simple squared error:

$$
L = (y-\hat{y})^2
$$

The derivative with respect to the prediction is:

$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y}-y)
$$

The derivative gives the direction and magnitude of the local change in loss with respect to the prediction. Backpropagation applies the chain rule to propagate this information through the network.

## 🧭 Loss Landscape

The loss landscape describes how the objective changes as model parameters change. Training tries to find parameter values with low loss, although non-convex deep learning objectives may contain saddle points, flat regions, and many local minima.

---

# 🔤 4. Important Terminology

----------------------------------------------------------------------- Term                                Explanation ----------------------------------- ----------------------------------- Target/label                        The correct output associated with an input

Prediction                          Output produced by the model

Logit                               Raw model score before a probability-producing activation

Probability                         A value interpreted as a probability, commonly between 0 and 1

Batch                               A subset of examples processed in one training step

Epoch                               One pass through the training dataset

Gradient                            Derivative of the objective with respect to parameters

Backpropagation                     Efficient calculation of gradients using the chain rule

Optimizer                           Algorithm that updates trainable parameters

Reduction                           How per-example losses are aggregated, such as mean or sum

Class imbalance                     A dataset where classes have substantially different frequencies

Label smoothing                     Replacing hard one-hot targets with softened target distributions

Robust loss                         A loss designed to be less sensitive to certain data errors or outliers

Calibration                         Agreement between predicted confidence and actual outcome frequencies -----------------------------------------------------------------------

## 📌 Important Distinction: Logits and Probabilities

For classification, models often produce logits. A softmax or sigmoid operation can transform logits into probabilities. Many library losses accept logits directly because they can combine activation and loss computation in a numerically stable way.

Do not apply softmax or sigmoid twice. Always check whether the chosen loss expects logits or probabilities.

---

# 🧮 5. Mathematical Foundations

## 5.1 Empirical Risk Minimization

The common training goal is to minimize average loss on the training data:

$$
\theta^* = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} L\left(y_i,f_\theta(x_i)\right)
$$

This is called empirical risk minimization. The goal is not simply to memorize the training set; the model should generalize to unseen examples.

## 5.2 Gradient Descent

A basic gradient descent update is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)
$$

Where:

- $$\theta_t$$ represents parameters at step $$t$$.
- $$\eta$$ is the learning rate.
- $$\nabla_\theta J$$ is the gradient of the objective.

The loss defines what is optimized; the optimizer defines how parameters are updated.

## 5.3 Batch Reduction

For per-example losses $$L_1,\ldots,L_N$$:

**Mean reduction**

$$
L_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N}L_i
$$

**Sum reduction**

$$
L_{\text{sum}} = \sum_{i=1}^{N}L_i
$$

Mean reduction generally keeps loss scale more stable as batch size changes. Sum reduction can make gradient magnitude depend strongly on batch size. Weighted and masked losses may use other normalization schemes.

## 5.4 Probabilistic Interpretation

Many losses correspond to likelihood-based estimation. Minimizing negative log-likelihood is equivalent to maximizing the likelihood of observed targets under the model's assumed probability distribution.

Examples:

- Gaussian observation model with fixed variance leads to a
squared-error objective, up to scaling and constants.
- Laplace observation model leads to an absolute-error objective, up
to scaling and constants.
- Bernoulli targets lead to binary cross-entropy.
- Categorical targets lead to categorical cross-entropy.

These relationships explain why the statistical assumptions behind a loss matter.

---

# 📚 6. Types of Loss Functions

# 6.1 📏 Regression Loss Functions

Regression predicts a continuous value such as price, temperature, distance, or demand.

## A. Mean Squared Error (MSE)

MSE squares the difference between actual and predicted values and averages the results.

$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

**Example:** If the actual values are $$[3,5]$$ and predictions are $$[2,7]$$:

$$
L_{\text{MSE}} = \frac{(3-2)^2+(5-7)^2}{2} = 2.5
$$

**Advantages** - Smooth and differentiable. - Strongly penalizes large errors. - Commonly effective when large errors should be expensive.

**Limitations** - Sensitive to outliers because errors are squared. - Units are squared, which can make interpretation less intuitive. - A few extreme errors can dominate the average.

**Use cases:** House-price prediction, sensor calibration, and regression tasks where large deviations deserve substantial penalties.

## B. Mean Absolute Error (MAE)

MAE averages absolute differences.

$$
L_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

**Advantages** - More robust to outliers than MSE. - Error remains in the same unit as the target. - Easy to interpret.

**Limitations** - Not differentiable at zero, although subgradients make it usable in modern frameworks. - Its gradient magnitude is constant away from zero, which may make optimization less smooth. - It can be less sensitive than MSE to large errors.

**Use cases:** Demand prediction, price forecasting, and data with occasional extreme values.

## C. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

$$
L_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}
$$

RMSE is expressed in the target's original units and penalizes large errors more than MAE. Because the square root changes gradient behavior, RMSE is not identical to MSE as an optimization objective.

## D. Huber Loss

Huber loss behaves quadratically for small errors and linearly for large errors.

$$
L_\delta(y,\hat{y}) =
\begin{cases}
\frac{1}{2}(y-\hat{y})^2, & |y-\hat{y}| \le \delta \\
\delta\left(|y-\hat{y}|-\frac{1}{2}\delta\right), & |y-\hat{y}| > \delta
\end{cases}
$$

Here, $$\delta$$ controls the transition point.

**Advantages** - Smooth near zero. - Less sensitive to extreme outliers than MSE. - Combines useful properties of squared and absolute error.

**Limitations** - Requires choosing an appropriate $$\delta$$. - Performance depends on the target scale and outlier distribution.

**Use cases:** Robotics, object localization, and regression datasets containing noisy labels.

## E. Log-Cosh Loss

Log-cosh is a smooth approximation to absolute error.

$$
L_{\text{log-cosh}} = \frac{1}{N}\sum_{i=1}^{N}\log\left(\cosh(\hat{y}_i-y_i)\right)
$$

For small errors, it behaves approximately quadratically; for large errors, it behaves approximately linearly.

**Use cases:** Robust regression where a smooth objective is useful.

## F. Mean Squared Logarithmic Error (MSLE)

MSLE compares logarithms of nonnegative targets and predictions.

$$
L_{\text{MSLE}} = \frac{1}{N}\sum_{i=1}^{N}\left[\log(1+y_i)-\log(1+\hat{y}_i)\right]^2
$$

**Use cases:** Nonnegative targets where relative differences matter, such as some growth or sales prediction tasks.

**Caution:** Targets and predictions must be in the valid domain; negative values are not directly supported by this formula.

---

# 6.2 🎯 Classification Loss Functions

Classification predicts a category or class probability.

## A. Binary Cross-Entropy (BCE)

Binary cross-entropy is used for binary classification, including independent multi-label outputs.

$$
L_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

Where $$y_i \in \{0,1\}$$ and $$p_i$$ is the predicted probability of the positive class.

For one example with target $$y=1$$ and predicted probability $$p=0.9$$:

$$
L = -\log(0.9) \approx 0.105
$$

If the model predicts $$p=0.1$$ for that same positive target:

$$
L = -\log(0.1) \approx 2.303
$$

A confident correct prediction has low loss; a confident incorrect prediction has high loss.

**Advantages** - Works naturally with probability-based binary classification. - Penalizes incorrect confident predictions. - Supports multi-label tasks by applying a binary objective to each label.

**Limitations** - Can be affected by class imbalance. - Very confident wrong predictions can contribute large losses. - The target encoding and output/loss pairing must be correct.

**Use cases:** Spam detection, fraud detection, disease classification, and independent tags such as image labels.

## B. Categorical Cross-Entropy

Categorical cross-entropy is commonly used for mutually exclusive multi-class classification.

$$
L_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\log(p_{ik})
$$

Where:

- $$K$$ is the number of classes.
- $$y_{ik}$$ is the target indicator or target probability for class
$$k$$.
- $$p_{ik}$$ is the predicted probability for class $$k$$.

With one-hot targets, only the true class contributes to the sum.

**Use cases:** Classifying an image as cat, dog, or bird; recognizing a digit; classifying a document into one category.

## C. Sparse Categorical Cross-Entropy

Sparse categorical cross-entropy has the same basic objective as categorical cross-entropy but accepts integer class labels instead of one-hot target vectors.

Example labels:

``` python
y_true = [2, 0, 1, 2]
```

Use it when each example belongs to exactly one class and the framework expects integer labels. Confirm whether the loss expects logits or probabilities.

## D. Softmax Cross-Entropy with Logits

For mutually exclusive multi-class classification, a model can output raw logits and use a numerically stable cross-entropy implementation that incorporates log-softmax internally.

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

Where $$z_k$$ is the logit for class $$k$$.

For a single example with target class $$t$$:

$$
L = -\log\left(\frac{e^{z_t}}{\sum_{j=1}^{K}e^{z_j}}\right)
$$

**Best practice:** Prefer a stable `from_logits=True` or equivalent loss when using raw logits, rather than manually applying softmax and then passing the result to a logits-based loss.

## E. Hinge Loss

Hinge loss is associated with maximum-margin classifiers, including support vector machines.

For binary labels $$y \in \{-1,+1\}$$ and score $$f(x)$$:

$$
L_{\text{hinge}} = \max(0,1-yf(x))
$$

It encourages the correct class score to exceed a margin.

**Advantages:** Margin-based interpretation and zero loss for examples that meet the margin.

**Limitations:** It is not a probability-calibration objective by itself and is less commonly used as the default for modern neural classification.

## F. Kullback-Leibler (KL) Divergence

KL divergence compares a target distribution $$P$$ with a predicted distribution $$Q$$.

$$
D_{\mathrm{KL}}(P\|Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}
$$

It measures how one probability distribution differs from another. It is asymmetric:

$$
D_{\mathrm{KL}}(P\|Q) \ne D_{\mathrm{KL}}(Q\|P)
$$

**Use cases:** Knowledge distillation, variational methods, distribution matching, and soft-label training.

**Caution:** Frameworks differ in the expected argument order and whether inputs should be probabilities or log-probabilities.

## G. Focal Loss

Focal loss down-weights well-classified examples so training can focus more on hard examples.

For binary classification:

$$
L_{\text{focal}} = -\alpha_t(1-p_t)^\gamma\log(p_t)
$$

Where:

- $$p_t$$ is the predicted probability of the target class.
- $$\alpha_t$$ balances class importance.
- $$\gamma \ge 0$$ controls how strongly easy examples are
down-weighted.

When $$\gamma=0$$, focal loss reduces to a class-weighted cross-entropy form.

**Use cases:** Object detection and severe class imbalance, such as detecting rare objects among many background examples.

**Limitations:** Extra hyperparameters can make tuning harder; focal loss does not automatically solve every imbalance or data-quality problem.

---

# 6.3 🖼️ Image Segmentation Losses

Segmentation predicts a class for each pixel.

## A. Dice Loss

Dice similarity measures overlap between a predicted mask and the target mask. A smoothed Dice loss can be written as:

$$
L_{\text{Dice}} = 1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}
$$

Where $$p_i$$ is the predicted foreground probability, $$y_i$$ is the target mask, and $$\epsilon$$ prevents division by zero.

**Advantages** - Directly encourages overlap. - Often useful when foreground pixels are rare.

**Limitations** - Batch-wise and per-image implementations can behave differently. - Smoothing and empty-mask handling matter. - It may be combined with cross-entropy for more stable optimization.

## B. Intersection over Union (IoU) / Jaccard Loss

IoU measures intersection relative to union.

$$
\operatorname{IoU} = \frac{|P \cap Y|}{|P \cup Y|}
$$

A simple IoU loss is:

$$
L_{\text{IoU}} = 1 - \operatorname{IoU}
$$

Soft variants use probabilities rather than hard masks to make the expression differentiable.

**Use cases:** Medical image segmentation, satellite imagery, and road-scene segmentation.

## C. Binary Cross-Entropy + Dice Loss

A combined objective can provide pixel-wise supervision and region-overlap optimization:

$$
L_{\text{combined}} = \lambda L_{\text{BCE}} + (1-\lambda)L_{\text{Dice}}
$$

The value of $$\lambda$$ controls the balance between the two terms.

**Important:** The two losses can have different numerical scales. Tune weights and monitor validation performance rather than assuming equal weights are always optimal.

---

# 6.4 📦 Object Detection Losses

Object detection often combines several objectives:

$$
L_{\text{total}} = \lambda_{\text{box}}L_{\text{box}} + \lambda_{\text{cls}}L_{\text{cls}} + \lambda_{\text{obj}}L_{\text{obj}}
$$

Depending on the detector, the components can include:

- **Box regression loss:** Measures bounding-box localization error.
- **Classification loss:** Measures incorrect class predictions.
- **Objectness/confidence loss:** Measures whether an object is
present at a location.
- **Distributional or auxiliary losses:** Used by particular detector
architectures.

### Bounding-Box Losses

----------------------------------------------------------------------- Loss                    Core idea               Typical role ----------------------- ----------------------- ----------------------- L1 / Smooth L1          Coordinate difference   Box coordinate regression

IoU loss                Overlap between boxes   Direct overlap optimization

GIoU                    Adds a penalty for the  Helps when boxes do not smallest enclosing box  overlap

DIoU                    Adds center-distance    Encourages closer box penalty                 centers

CIoU                    Adds center-distance    More complete geometric and aspect-ratio terms  box objective -----------------------------------------------------------------------

The exact loss composition varies by model and version. Always inspect the implementation or documentation for the detector being used.

---

# 6.5 🧠 Metric Learning and Representation Losses

These losses help learn useful embeddings, often so similar examples are close and dissimilar examples are far apart.

## A. Contrastive Loss

A common contrastive objective for a pair of embeddings is:

$$
L = yD^2 + (1-y)\max(0,m-D)^2
$$

In this convention, $$y=1$$ means a similar pair, $$D$$ is embedding distance, and $$m$$ is the margin for dissimilar pairs. Other implementations reverse the label convention, so verify the definition before using the formula.

## B. Triplet Loss

Triplet loss uses an anchor, a positive example, and a negative example.

$$
L_{\text{triplet}} = \max\left(0,D(a,p)-D(a,n)+m\right)
$$

Where:

- $$a$$ is the anchor.
- $$p$$ is a similar/positive example.
- $$n$$ is a dissimilar/negative example.
- $$m$$ is the required margin.

**Use cases:** Face recognition, image retrieval, product matching, and identity verification.

**Limitation:** Effective triplet mining and carefully constructed batches are important. Many triplets produce zero loss and no useful gradient.

## C. Cosine Embedding Loss

This loss encourages embeddings to have appropriate cosine similarity according to pair labels. It is useful when angular similarity matters more than vector magnitude.

---

# 6.6 🎭 Generative and Self-Supervised Losses

----------------------------------------------------------------------- Loss                    Purpose                 Example application ----------------------- ----------------------- ----------------------- Reconstruction loss     Compare input with      Autoencoders reconstructed output

Perceptual loss         Compare feature         Image super-resolution representations rather than only pixels

Adversarial loss        Train a generator       GANs against a discriminator

KL divergence term      Regularize latent       Variational distributions           autoencoders

InfoNCE / contrastive   Separate positive and   Self-supervised objective               negative pairs in       learning representation space

Diffusion denoising     Predict noise or        Diffusion models loss                    another training target at sampled timesteps -----------------------------------------------------------------------

### Variational Autoencoder (VAE) Objective

A common VAE objective combines reconstruction loss and KL divergence:

$$
L_{\text{VAE}} = L_{\text{reconstruction}} + \beta D_{\mathrm{KL}}\left(q_\phi(z|x)\|p(z)\right)
$$

The parameter $$\beta$$ controls the strength of latent-distribution regularization in a beta-VAE-style objective.

### Generative Adversarial Networks (GANs)

GANs use objectives for both the generator and discriminator. The original minimax formulation is not the only option; practical GANs may use non-saturating, hinge, Wasserstein, or other objectives.

---

# 📊 7. Loss Functions by Task

---

Task                     Common starting point   Important considerations ------------------------ ----------------------- ----------------------- Binary classification    Binary cross-entropy    Use logits-aware loss when model outputs raw scores

Multi-class, one class   Categorical or sparse   Softmax/logits and per example              categorical             label format must match cross-entropy

Multi-label              Binary cross-entropy    Labels are not mutually classification           per label               exclusive

Regression               MSE or MAE              Choose based on outliers and error cost

Robust regression        Huber or log-cosh       Tune scale/transition parameters

Imbalanced               Weighted BCE/CE or      Also inspect sampling, classification           focal loss              data quality, and per-class metrics

Semantic segmentation    CE, Dice, IoU, or a     Foreground imbalance combination             and empty masks

Object detection         Model-specific box,     Use the detector's class, and objectness   prescribed formulation losses

Face recognition         Triplet or              Pair/triplet contrastive-style       construction is objective               critical

Autoencoder              Reconstruction loss     Match loss to data type and output activation

VAE                      Reconstruction plus KL  Balance reconstruction term                    and latent regularization

Knowledge distillation   Soft-target CE or       Temperature and KL-based objective      reduction conventions matter

Ranking/recommendation   Pairwise or listwise    Match the objective to ranking loss            ranking quality
---

---

# 🆚 8. Comparison of Common Loss Functions

## 8.1 Regression Loss Comparison

--------------------------------------------------------------------------------- Loss           Error penalty   Outlier        Differentiability   Best fit sensitivity -------------- --------------- -------------- ------------------- --------------- MSE            Quadratic       High           Smooth              Large errors should be strongly penalized

MAE            Linear          Lower          Not differentiable  Robust, at zero             interpretable absolute error

RMSE           Square root of  High           Smooth except       Reporting error average squared                special edge cases  in target units error

Huber          Quadratic near  Moderate       Smooth at           Noisy zero, linear                   transition with     regression for large                      standard definition errors

Log-cosh       Smooth,         Lower than MSE Smooth              Robust approximately                                      optimization quadratic then linear ---------------------------------------------------------------------------------

## 8.2 Classification Loss Comparison

------------------------------------------------------------------------------------------------------ Loss            Typical task               Uses probabilities?        Main strength   Main caution --------------- -------------------------- -------------------------- --------------- ---------------- Binary          Binary / multi-label       Probability form or        Standard        Class imbalance cross-entropy                              logits-aware form          probabilistic objective

Categorical     Multi-class                Probability form or        Penalizes low   Label/output cross-entropy                              logits-aware form          probability for format mismatch the true class

Sparse          Multi-class integer labels Implementation-dependent   Avoids one-hot  Integer class categorical CE                             logits/probability input   encoding        IDs must be valid

Hinge           Margin-based               No probability requirement Margin          Not inherently classification                                        optimization    calibrated

Focal loss      Imbalanced                 Commonly                   Focuses on hard Hyperparameter classification/detection   probability/logits-based   examples        sensitivity implementations

KL divergence   Distribution matching      Distribution or            Compares        Asymmetric; log-probability form       probability     argument distributions   convention matters ------------------------------------------------------------------------------------------------------

## 8.3 Choosing a Loss: Decision Flow

``` mermaid
flowchart TD
    A[Identify the prediction task] --> B{What is the target?}
    B -->|Continuous value| C{Are outliers important?}
    C -->|Large errors need strong penalty| D[MSE]
    C -->|Want more robustness| E[MAE or Huber]
    B -->|One class per example| F[Cross-entropy]
    B -->|Multiple independent labels| G[Binary cross-entropy per label]
    B -->|Pixel mask| H[Cross-entropy, Dice, IoU, or combination]
    B -->|Bounding boxes| I[Detector-specific box and classification losses]
    B -->|Embedding similarity| J[Contrastive or triplet objective]
    B -->|Generated/reconstructed data| K[Reconstruction or generative objective]
```

---

# 💻 9. Practical Python Examples

## 9.1 Calculate Losses with NumPy

``` python
import numpy as np

y_true = np.array([3.0, 5.0, 2.0, 8.0])
y_pred = np.array([2.5, 5.5, 4.0, 7.0])

mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(mse)

print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
```

**Explanation:** The same predictions can produce different loss values because each function penalizes errors differently.

## 9.2 Binary Cross-Entropy with NumPy

``` python
import numpy as np

y_true = np.array([1, 0, 1, 1], dtype=float)
y_prob = np.array([0.90, 0.20, 0.65, 0.40], dtype=float)

# Clip probabilities to avoid log(0).
eps = 1e-7
y_prob = np.clip(y_prob, eps, 1 - eps)

bce = -np.mean(
    y_true * np.log(y_prob)
    + (1 - y_true) * np.log(1 - y_prob)
)

print(f"Binary cross-entropy: {bce:.4f}")
```

**Explanation:** Clipping protects this educational probability-based implementation from undefined logarithms. In production training, prefer a numerically stable framework loss that accepts logits when appropriate.

## 9.3 TensorFlow / Keras: Regression

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(10,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Linear output for regression
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

# X_train shape: (number_of_samples, 10)
# y_train shape: (number_of_samples,) or (number_of_samples, 1)
# model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32)
```

**Explanation:** MSE is the training objective; MAE is tracked as an additional metric. The final layer has no activation because regression targets are assumed to be unrestricted real values.

## 9.4 TensorFlow / Keras: Multi-Class Classification

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_features = 20
num_classes = 4

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(num_classes)  # Raw logits
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# y_train contains integer class IDs: 0, 1, 2, or 3.
# model.fit(X_train, y_train, validation_split=0.2, epochs=15)
```

**Explanation:** `from_logits=True` tells the loss to interpret the final layer's values as raw logits. Do not add a softmax layer to this model unless you also adjust the loss configuration accordingly.

## 9.5 TensorFlow / Keras: Multi-Label Classification

``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_features = 30
num_labels = 5

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_labels)  # One logit per independent label
])

model.compile(
    optimizer="adam",
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.BinaryAccuracy(
            threshold=0.0  # Appropriate for logits
        )
    ]
)
```

**Explanation:** Multi-label tasks allow more than one label to be true at once. Each output is treated as a separate binary decision. A softmax is generally inappropriate when labels are independent.

## 9.6 PyTorch: Regression Losses

``` python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, 5.0, 2.0, 8.0])
y_pred = torch.tensor([2.5, 5.5, 4.0, 7.0])

mse_loss = nn.MSELoss()(y_pred, y_true)
mae_loss = nn.L1Loss()(y_pred, y_true)
huber_loss = nn.HuberLoss(delta=1.0)(y_pred, y_true)

print("MSE:", mse_loss.item())
print("MAE:", mae_loss.item())
print("Huber:", huber_loss.item())
```

## 9.7 PyTorch: Multi-Class Classification with Logits

``` python
import torch
import torch.nn as nn

# Shape: (batch_size, number_of_classes)
logits = torch.tensor([
    [2.5, 0.3, -0.4],
    [0.1, 1.8, 0.2]
])

# Shape: (batch_size,), integer class IDs
targets = torch.tensor([0, 1])

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)

print("Cross-entropy loss:", loss.item())
```

**Important:** `nn.CrossEntropyLoss` expects raw logits and integer class indices for this common usage. Do not apply softmax first.

## 9.8 PyTorch: Binary Classification with Logits

``` python
import torch
import torch.nn as nn

logits = torch.tensor([1.5, -0.7, 0.2])
targets = torch.tensor([1.0, 0.0, 1.0])

criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)

print("Binary loss:", loss.item())
```

`BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable way.

## 9.9 Custom Training Loop Concept

``` mermaid
flowchart TD
    A[Get batch] --> B[Clear old gradients]
    B --> C[Forward pass]
    C --> D[Compute loss]
    D --> E[Backpropagate]
    E --> F[Optimizer step]
    F --> G[Log loss and metrics]
    G --> A
```

A typical PyTorch training step looks like:

``` python
optimizer.zero_grad()
predictions = model(inputs)
loss = criterion(predictions, targets)
loss.backward()
optimizer.step()
```

This is a conceptual template. `criterion`, target shapes, output dimensions, and activation/loss pairing must match the task.

---

# 🧪 10. Mini-Project: Compare Regression Losses

## 🎯 Objective

Create a small experiment showing how MSE, MAE, and Huber loss react to an outlier. This helps build intuition for choosing a loss based on the data and business requirement.

## 🪜 Project Steps

1. Create actual target values and predictions.
2. Add one deliberately large prediction error.
3. Calculate MSE, MAE, and Huber loss.
4. Compare the results.
5. Repeat after removing the outlier.
6. Explain which objective best matches the application's error costs.

## 💻 Complete Example

``` python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(error)

    per_example_loss = np.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta)
    )
    return np.mean(per_example_loss)

y_true = np.array([10, 12, 13, 15, 16], dtype=float)

# One large error is intentionally included.
y_pred = np.array([10.5, 11.5, 13.5, 14.5, 40], dtype=float)

mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))
huber = huber_loss(y_true, y_pred, delta=1.0)

print(f"MSE:   {mse:.3f}")
print(f"MAE:   {mae:.3f}")
print(f"Huber: {huber:.3f}")
```

## 📊 Expected Interpretation

- **MSE:** The large error contributes disproportionately because it
is squared.
- **MAE:** The large error matters linearly, so it is less dominant
than under MSE.
- **Huber:** Small errors are treated quadratically while large errors
are treated approximately linearly.
- There is no universally best loss. If a large error is genuinely
costly, MSE may be desirable. If it is a data anomaly, a robust loss may be more appropriate.

## 📝 Suggested Extension

- Plot each per-example loss against absolute error.
- Try several Huber `delta` values.
- Normalize or transform the target and observe how the loss scale
changes.
- Record the metric used for model selection separately from the
training loss.

---

# 🌍 11. Real-World Use Cases

## 🏠 House Price Prediction

MSE is a common baseline when large price errors should be penalized strongly. MAE is useful when the average absolute difference in currency units is easier to explain to stakeholders.

## 🩺 Medical Image Classification

Cross-entropy is a common starting point for mutually exclusive disease classes. If rare diseases are underrepresented, consider class weights, careful sampling, and per-class recall. Loss optimization does not replace clinical validation.

## 🚗 Autonomous Driving and Road Inspection

Object detectors may combine classification and bounding-box losses. For a pothole detection system, evaluate not only the aggregate loss but also precision, recall, false alarms, missed potholes, localization quality, and performance across weather and lighting conditions.

## 🛒 Recommendation Systems

Pairwise ranking losses can teach a model to score a relevant item above an irrelevant one. The training objective should align with the intended ranking behavior and the evaluation protocol.

## 🗣️ Language Models

Token-level cross-entropy is commonly used to train a model to predict the next token. The loss is generally aggregated over valid target tokens, with padding tokens masked out.

## 🖼️ Image Segmentation

Cross-entropy provides pixel-wise classification supervision, while Dice or IoU-related losses encourage overlap. Combining them can be useful when the foreground occupies a small part of an image.

## 🧬 Representation Learning

Contrastive and triplet losses learn embeddings that preserve useful similarity relationships. Data sampling and the definition of positive and negative pairs strongly affect the result.

---

# 🚀 12. Advanced Concepts

## 12.1 Class-Weighted Loss

When some classes are more important or less frequent, a weighted objective can give their errors greater influence.

For class weights $$w_k$$:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}w_{y_i}\log(p_{i,y_i})
$$

Weights should be chosen carefully. Excessive weighting can increase false positives or destabilize training. Always inspect per-class precision and recall.

## 12.2 Masked Loss

Some tasks contain invalid, missing, padded, or ignored targets. A mask $$m_i$$ can exclude those entries:

$$
L_{\text{masked}} = \frac{\sum_i m_i L_i}{\sum_i m_i}
$$

If the mask is entirely zero, the implementation must handle the denominator safely. Masking should exclude only genuinely invalid targets, not difficult examples that the model should learn from.

## 12.3 Label Smoothing

For a classification target with $$K$$ classes, label smoothing distributes some target probability away from the hard target. One common convention is:

$$
y'_k = (1-\epsilon)y_k + \frac{\epsilon}{K}
$$

Where $$\epsilon$$ is a small smoothing factor. Implementations may use slightly different conventions.

**Potential benefits:** Reduced overconfidence and improved generalization in some settings.

**Potential drawbacks:** Too much smoothing can weaken supervision, especially when labels are reliable or classes are fine-grained.

## 12.4 Multi-Task Learning

A model may predict multiple targets, such as object class, bounding box, and depth. A combined loss is:

$$
L_{\text{total}} = \sum_{j=1}^{M}\lambda_j L_j
$$

Each $$\lambda_j$$ controls the influence of task $$j$$.

Loss terms may have different scales, gradient magnitudes, and difficulty. A task with a numerically large loss can dominate even when it is not the most important task. Inspect each component and consider principled weighting methods.

## 12.5 Dynamic and Uncertainty-Based Loss Weighting

Instead of manually fixing all task weights, some multi-task methods learn task-dependent weights or adapt weights during training. These approaches can help when tasks have different noise levels, but they introduce assumptions and extra implementation complexity.

## 12.6 Curriculum Learning and Hard-Example Mining

- **Curriculum learning:** Starts with easier examples or simpler
tasks and gradually introduces harder ones.
- **Hard-example mining:** Gives more attention to examples the model
currently handles poorly.
- **Focal loss:** A loss-level mechanism that reduces the contribution
of well-classified examples.

These methods are related but not interchangeable. Hard examples may be informative, mislabeled, ambiguous, or out-of-distribution; inspect them rather than assuming all hard examples deserve more weight.

## 12.7 Loss Scaling and Mixed Precision

Mixed-precision training can improve speed and memory efficiency. Some low-precision gradients may underflow, so frameworks may use loss scaling. Loss scaling changes numerical representation during gradient computation; it does not change the conceptual learning objective when handled correctly.

## 12.8 Numerical Stability

Common safeguards include:

- Use logits-aware cross-entropy implementations.
- Avoid manually computing `log(0)`.
- Use stable `logsumexp` or framework equivalents.
- Check for NaN and infinite loss values.
- Keep target and prediction shapes aligned.
- Use appropriate floating-point precision.
- Verify that masked or weighted reductions have safe denominators.

## 12.9 Training Loss vs. Validation Loss

Training loss measures fit to training examples. Validation loss measures performance on held-out data used for model selection.

``` mermaid
flowchart LR
    A[Training loss decreases] --> B{Validation loss}
    B -->|Also decreases| C[Generalization may be improving]
    B -->|Plateaus| D[Consider tuning or early stopping]
    B -->|Increases repeatedly| E[Possible overfitting]
```

A growing gap between training and validation loss can indicate overfitting, but data distribution shifts, noisy labels, regularization, and different training/validation processing can also affect the curves.

## 12.10 Loss Is Not the Same as Business Success

A lower training loss does not guarantee a better real-world model. Select and monitor evaluation measures that match the application, such as recall at a chosen false-positive rate, calibration, latency, robustness, or ranking quality.

---

# ✅ 13. Best Practices

1. **Match the loss to the task.** Determine whether the output is
continuous, binary, multi-class, multi-label, a mask, a box, or an embedding.
2. **Match the loss to the output representation.** Check whether it
expects logits, probabilities, or scores.
3. **Match labels to the loss.** One-hot vectors, integer class IDs,
binary vectors, and continuous targets are not interchangeable.
4. **Start with a reliable baseline.** MSE/MAE for regression and
cross-entropy for classification are useful starting points.
5. **Inspect the target distribution.** Check skew, outliers, label
noise, class imbalance, and missing values.
6. **Track task-relevant metrics.** Classification accuracy alone can
be misleading on imbalanced datasets.
7. **Inspect per-class results.** Use confusion matrices, class-level
precision and recall, and representative error examples.
8. **Monitor both training and validation loss.** Do not select a model
using training loss alone.
9. **Tune weighting carefully.** Class weights and multi-task weights
can improve one behavior while harming another.
10. **Use numerically stable built-in losses.** Especially for
cross-entropy and log-based objectives.
11. **Check reduction behavior.** Mean, sum, masking, and weighting
change gradient scale.
12. **Set random seeds and record configurations.** Reproducibility
matters when comparing loss functions.
13. **Evaluate on untouched test data only after model selection.**
Avoid repeatedly tuning against the test set.
14. **Validate real-world costs.** The best loss is the one that
supports the intended outcome, not necessarily the most familiar formula.

---

# ⚠️ 14. Common Mistakes

----------------------------------------------------------------------- Mistake                 Why it is a problem     Better practice ----------------------- ----------------------- ----------------------- Applying softmax before Applies the             Pass raw logits to the a logits-based          transformation twice or logits-aware loss cross-entropy loss      violates the loss input contract

Using categorical CE    Shape/type mismatch or  Use sparse CE for with the wrong target   incorrect objective     integer IDs or format                                          categorical CE for one-hot/distribution targets, as required

Using softmax for       Forces class            Use independent independent multi-label probabilities to        sigmoid-style outputs targets                 compete and sum to one  with BCE

Using MSE for every     May not match target    Select based on task problem                 distribution or error   and statistical costs                   assumptions

Ignoring outliers       A few observations may  Investigate outliers dominate MSE            and consider MAE/Huber if appropriate

Treating accuracy as    Accuracy is usually not Train with a the loss                a useful differentiable differentiable loss and objective               evaluate accuracy separately

Comparing raw losses    Their scales and        Compare task metrics from different          meanings differ         and interpret each loss objectives as if                                in context directly equivalent

Ignoring class          Aggregate loss/accuracy Use class-level metrics imbalance               may hide poor           and consider weighting minority-class          or sampling performance

Wrong tensor shapes     Broadcasting can        Check shapes and dtypes silently produce an     before training unintended calculation

Wrong label range       Invalid class IDs can   Validate targets cause errors or         against the number of incorrect results       classes

Ignoring loss reduction Gradient scale can      Understand mean/sum and change with batch size  weighted reduction semantics

Overweighting rare      Can lead to too many    Tune weights on classes                 false positives         validation data

Tuning on test data     Makes test results      Use training/validation optimistic and          for decisions; test unreliable              once at the end

Assuming lower loss     Loss may not reflect    Track always means better     safety, calibration, or application-specific deployment performance  business requirements   metrics and robustness -----------------------------------------------------------------------

---

# 🎤 15. Interview Questions and Points

## Q1. What is a loss function?

A loss function measures the mismatch between model predictions and target values. Its gradients guide parameter updates during training.

## Q2. What is the difference between a loss function and an evaluation metric?

A loss is the objective optimized during training. A metric measures performance for interpretation or model selection and may not be differentiable.

## Q3. Why is MSE sensitive to outliers?

MSE squares errors, so large deviations contribute disproportionately to the total loss.

## Q4. What is the difference between MAE and MSE?

MAE penalizes errors linearly and is more robust to outliers. MSE penalizes errors quadratically and emphasizes large errors more strongly.

## Q5. When would you use Huber loss?

When small errors should receive a smooth quadratic penalty but large errors should not dominate as strongly as they would under MSE.

## Q6. Why is cross-entropy popular for classification?

It is tied to maximum-likelihood estimation for categorical or Bernoulli targets and strongly penalizes assigning low probability to the true class.

## Q7. What is the difference between binary and categorical cross-entropy?

Binary cross-entropy handles a binary target or independent labels. Categorical cross-entropy handles mutually exclusive classes represented by a class distribution; sparse categorical CE commonly accepts integer class IDs.

## Q8. Why do frameworks provide cross-entropy losses that accept logits?

They can combine the necessary activation and logarithmic operations in a numerically stable way.

## Q9. What is focal loss?

A loss that down-weights well-classified examples and focuses more on hard examples, often used in imbalanced detection and classification.

## Q10. What is Dice loss used for?

It encourages overlap between predicted and target regions and is commonly used for segmentation, especially with imbalanced foreground/background pixels.

## Q11. What is KL divergence?

A directional measure of difference between probability distributions, used in distribution matching, variational learning, and knowledge distillation.

## Q12. Why can training loss decrease while validation loss increases?

The model may be overfitting training data or the validation distribution may differ. The pattern should be investigated rather than interpreted in isolation.

## Q13. Can a loss function be negative?

Yes, depending on the definition and terms involved. Some common losses such as MSE and cross-entropy are nonnegative under their standard assumptions, while other objectives or likelihood-based expressions can take negative values.

## Q14. What is a composite loss?

A weighted combination of multiple objectives, such as classification plus box regression or reconstruction plus KL divergence.

## Q15. How do you choose a loss for an imbalanced dataset?

Start by understanding the task and error costs. Consider class weighting, focal loss, sampling, and appropriate metrics, then validate the trade-offs. No single technique works best in every case.

## Q16. What is label smoothing?

A technique that softens hard class targets by distributing a small amount of target probability across classes.

## Q17. Why is the reduction method important?

It determines how per-example losses are aggregated. Mean and sum reductions can produce different gradient scales, particularly when batch sizes vary.

## Q18. Is lower loss always better?

Only when comparing the same objective under comparable conditions. Lower loss does not necessarily imply better calibration, robustness, or application-level performance.

## 🧠 Interview Formula Sheet

--------------------------------------------------------------------------------------------- Concept                             Formula ----------------------------------- --------------------------------------------------------- MSE                                 $$\frac{1}{N}\sum_i(y_i-\hat{y}_i)^2$$

MAE                                 $$\frac{1}{N}\sum_i|y_i-\hat{y}_i|$$

Binary cross-entropy                $$-\frac{1}{N}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$$

Categorical cross-entropy           $$-\frac{1}{N}\sum_i\sum_k y_{ik}\log p_{ik}$$

Hinge loss                          $$\max(0,1-yf(x))$$

KL divergence                       $$\sum_iP(i)\log\frac{P(i)}{Q(i)}$$

Gradient descent                    $$\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)$$ ---------------------------------------------------------------------------------------------

---

# ⚡ 16. Quick Revision

## 🔑 Key Points

- A loss function quantifies prediction error and defines the training
objective.
- Backpropagation computes gradients of the loss; the optimizer uses
them to update parameters.
- **Regression:** MSE emphasizes large errors; MAE is more robust to
outliers; Huber combines quadratic and linear behavior.
- **Binary classification:** Binary cross-entropy is a common
baseline.
- **Multi-class classification:** Categorical or sparse categorical
cross-entropy is common.
- **Multi-label classification:** Use an independent binary objective
for each label.
- **Segmentation:** Cross-entropy, Dice, IoU, and combinations are
common.
- **Object detection:** Losses usually combine classification and
localization terms.
- **Metric learning:** Contrastive and triplet losses organize
embedding spaces.
- **Advanced learning:** KL divergence, focal loss, label smoothing,
masking, and multi-task weighting address specific needs.
- A lower loss does not guarantee better real-world performance.
- Always verify logits/probabilities, target formats, tensor shapes,
reduction, and task-specific evaluation metrics.

## 🧾 Important Formulas

### Mean Squared Error

$$
L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

### Mean Absolute Error

$$
L_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

### Binary Cross-Entropy

$$
L_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

### Categorical Cross-Entropy

$$
L_{\text{CCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}\log(p_{ik})
$$

### Gradient Descent Update

$$
\theta_{t+1} = \theta_t - \eta\nabla_\theta J(\theta_t)
$$

## 🗺️ Visual Learning Roadmap

``` mermaid
flowchart TD
    A[Understand prediction task] --> B[Identify target format]
    B --> C[Choose a suitable loss]
    C --> D[Match activation and loss inputs]
    D --> E[Check shapes, labels, and reduction]
    E --> F[Train with backpropagation]
    F --> G[Monitor training and validation loss]
    G --> H[Evaluate task-specific metrics]
    H --> I[Inspect errors and tune responsibly]
```

## 🧠 One-Minute Summary

A neural network makes predictions, a loss function measures how wrong they are, backpropagation calculates how the loss changes with the parameters, and an optimizer updates those parameters. The correct loss depends on the task, target distribution, data quality, and real-world cost of errors. Begin with a well-established baseline, verify the model/loss interface, monitor validation behavior, and choose the final model using metrics that reflect the intended application.

---

## 📚 Further Learning

- TensorFlow/Keras loss functions:
https://www.tensorflow.org/api_docs/python/tf/keras/losses
- PyTorch loss functions:
https://pytorch.org/docs/stable/nn.html#loss-functions
- Scikit-learn model evaluation:
https://scikit-learn.org/stable/modules/model_evaluation.html

> **Final reminder:** A loss function is not just a formula to memorize.
> Understand what errors it penalizes, which assumptions it makes, and
> whether those priorities match your problem.
