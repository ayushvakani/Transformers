# Transformer Architecture from Scratch

> A from-scratch implementation of the Transformer model based on the seminal paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), built with Python, PyTorch, and Keras.

---

## Overview

This project was built with the goal of deeply understanding the internal mechanics of the Transformer architecture — one of the most influential models in modern deep learning. Rather than relying on high-level abstractions, every component is implemented from the ground up.

The focus is on the **decoder** and **encoder** stack, covering all the key components that power sequence generation.

---

## Architecture

### Components Implemented

- **Multi-Head Self-Attention** — Jointly attends to information from different representation subspaces at different positions.
- **Masked Multi-Head Attention** — Prevents positions from attending to subsequent positions, ensuring the autoregressive property during decoding.
- **Positional Encoding** — Injects sequence order information into token embeddings using sine and cosine functions.
- **Feed-Forward Layers** — Position-wise fully connected layers applied independently to each token.
- **Embeddings** — Token embedding layer that maps discrete tokens into continuous vector space.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| PyTorch | Tensor operations & autograd |
| Keras (TensorFlow backend) | High-level layer abstractions & model building |
| NumPy | Numerical computations & array operations |
| Matplotlib | Visualizing attention weights & training curves |

---

## Key Concepts Explored

- Attention mechanisms (scaled dot-product & multi-head)
- Sequence-to-sequence modeling
- Autoregressive decoding with masking
- Positional encoding strategies
- Layer normalization and residual connections
- Embeddings and representation learning


## Requirements

```txt
torch
tensorflow
keras
numpy
matplotlib
```


## Motivation

Large language models and modern AI systems are built on top of the Transformer. This project was designed to go beyond surface-level API usage and build a solid foundation in:

- How attention scores are computed and scaled
- Why masking is essential in decoder-only models
- How positional encodings preserve sequence order without recurrence
- The flow of data through a full decoder block

---

## References

- Vaswani et al. (2017) — [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)

---

> *Built for learning. Every line written with intent.*
