# 🧠 Multi-Layer Perceptron (MLP) Notation

Professional notes explaining the notation and dimensions used in a Multi-Layer Perceptron (MLP), with a visual architecture reference and worked parameter-count example.

## 📌 Overview

These notes cover:

- Dataset and layer notation
- Weight matrices and bias vectors
- Forward propagation equations
- Single-neuron notation
- Activation functions
- Matrix shapes and trainable parameter counts
- A worked example of a network with 26 parameters

## 🖼️ MLP Notation Diagram

![Multi-Layer Perceptron (MLP) Notation](MLP_Notation.png)

## 📐 Core Formulas

### Forward propagation

$$
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}
$$

$$
A^{(l)} = g^{(l)}\left(Z^{(l)}\right)
$$

### Weight and bias shapes

$$
W^{(l)} \in \mathbb{R}^{n_{l-1}\times n_l}
$$

$$
b^{(l)} \in \mathbb{R}^{n_l\times 1}
$$

### Total trainable parameters

$$
P_{\text{total}} = \sum_{l=1}^{L}(n_{l-1}+1)n_l
$$

For the example architecture in the diagram, the total is **26 trainable parameters**.

## 📂 Files

- [`MLP_Notation.md`](MLP_Notation.md) — detailed notes and formula reference
- [`MLP_Notation.png`](MLP_Notation.png) — MLP notation infographic


