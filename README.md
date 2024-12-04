# Multi-Head Attention Implementation in PyTorch

This repository contains an implementation of the **Multi-Head Attention** mechanism using PyTorch. The Multi-Head Attention is a core component of the Transformer architecture, which allows the model to jointly attend to information from different representation subspaces at different positions.

## Overview

- The implementation defines two main classes:
  1. **Attention**: A single attention head that computes scaled dot-product attention.
  2. **MultiHeadAttention**: A wrapper around multiple Attention heads, which computes multi-head attention by applying several attention mechanisms in parallel.

The model supports the usage of a mask for padding or other sequence manipulation tasks.

## File Structure

- `multi_head_attention.py`: Contains the `MultiHeadAttention` class.
- `self_attention.py`: Contains the `Attention` class.
## Requirements
- Python 3.x
- PyTorch
To install PyTorch, use the following command:

`
pip install torch
`
## Clone the Repository

To clone this repository, run the following command in your terminal:

`
git clone https://github.com/maturyusuf/Multihead_Self_Attention_from_Scratch.git
`
