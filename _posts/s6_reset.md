---
layout: post
title: Associative Boundary Resetting for SSMs (Mamba)
date: 2024-02-21
description: How to enforce boundary resetting for selective SSMs 
tags: ML
---

This post is about a technical detail of boundary resetting for Selective State Space
Models (S6) like [Mamba](https://arxiv.org/abs/2312.00752). I discuss boundary resetting
for Selective SSMs, and give an implementaiton that enables resetting at boundaries
without sacrificing the associativity of the binary scan operation, thereby maintaining
parallelizability.

This model architecture was introduced by Albert Gu, Tri Dao in the paper "Mamba:
Linear-Time Sequence Modeling with Selective State Spaces" and has gained attention as a
potential alternative architecture to transformers.

### Sequence Packing

When training large-scale sequence models like as transformer language models, hardware
efficiency during training is critical to acheiving strong results. A well-known trick
to improve harware utilization of LM training algorithms is sequence packing:
concatenating multiple seuqence into the same token array. Without sequence packing,
training is inefficnet due to abundance of padding tokens which contribute no
information to the loss, and its typcial to reduce padding token rates to well below 1%
with a bin-packing algorithm, prior to running 

The transformer architecture admits a simple mechanism to prevent sequence packed in to
the same array from interacting with eachother: a block-diagonal attention mask.
However, for non-attention based sequence models such as SSMs, alternative approaches
are requried. Let's look at how they deal with this in the Mamba paper

Section 3.5.2 "Boundary Resetting" reads:

> In settings where multiple independent sequences are stitched together, Transformers
> can keep them separate by instantiating a particular attention mask, while LTI models
> will bleed information between the sequences. Selective SSMs can also reset their
> state at boundaries (e.g. ∆𝑡 → ∞ or Theorem 1 when 𝑔𝑡 → 1). These settings may occur
> artificially (e.g. packing documents together to improve hardware utilization) or
> naturally (e.g. episode boundaries in reinforcement learning (Lu et al. 2023)).

I think its important to note from this that boundary resetting in Selective SSMs must
be *learned* the model. Recall that ∆𝑡 is the 


In this post I consider what it would mean to enforce boundary resetting in Selective
SSMs. To start, let's consider the simple case where 

We'll first demonstrate the concept using `jax.lax.scan` which is easier to reason
about, but extremely inefficient because it runs sequentially over the input instead of
using an efficnet parallel scan. Then we'll move onto a parallelized implementation
using `jax.lax.associative_scan`.


```python
import jax
import jax.numpy as jnp
from jaxtyping import Float, Bool, Array

def ssm(
    x:  Float[Array, "L D  "],
    dt: Float[Array, "L D  "],
    A:  Float[Array, "  D N"],
    B:  Float[Array, "L   N"],
    C:  Float[Array, "L   N"],
    D:  Float[Array, "  D  "],
) -> Float[Array, "L D"]:
    """Selective Scan operation."""
    l, d = x.shape
 
    # Discretize the continuous-time SSM (implementation omitted). See paper.
    dA, dB = discretize(A, B, dt)

    def scan_op(
        h: Float[Array, "D N"],
        params: tuple[
            Float[Array, "D  "],  # x
            Float[Array, "D N"],  # dA
            Float[Array, "D N"],  # dB
            Float[Array, "  N"],  # C
        ],
    ) -> tuple[Float[Array, "d n"], Float[Array, "d"]]:
        xi, dAi, dBi, Ci, reset = params
        h_ = dAi * h + dBi * xi[:, None]
        y = h_ @ Ci
        return h_, y

    h0 = jnp.zeros(shape=(d, n), dtype=x.dtype)
    _, y = jax.lax.scan(scan_op, h0, (x, dA, dB, C))
    return y + x * D
```

Now, we can enforce boundary resetting by conditionally resettting the hidden state to a
vector of zeros in the case when 

```python
def ssm(
    x:  Float[Array, "L D  "],
    dt: Float[Array, "L D  "],
    A:  Float[Array, "  D N"],
    B:  Float[Array, "L   N"],
    C:  Float[Array, "L   N"],
    D:  Float[Array, "  D  "],
    resets: Bool[Array, "L"],
) -> Float[Array, "L D"]:
    """Selective Scan operation."""
    l, d = x.shape
 
    # Discretize the continuous-time SSM (implementation omitted). See paper.
    dA, dB = discretize(A, B, dt)

    def scan_op(
        h: Float[Array, "D N"],
        params: tuple[
            Float[Array, "D  "],  # x
            Float[Array, "D N"],  # dA
            Float[Array, "D N"],  # dB
            Float[Array, "  N"],  # C
            Bool[Array,  ""],     # reset
        ],
    ) -> tuple[Float[Array, "d n"], Float[Array, "d"]]:
        xi, dAi, dBi, Ci, reset = params
        h_: = jax.lax.cond(
            reset,
            lambda _: jnp.zeros_like(h),
            lambda _: dAi * h + dBi * xi[:, None],
            None,
        )
        y = h_ @ Ci
        return h_, y

    h0 = jnp.zeros(shape=(d, n), dtype=x.dtype)
    _, y = jax.lax.scan(scan_op, h0, (x, dA, dB, C))
    return y + x * D
```


## Associative Boundary Resetting

A critical aspect of Mamba is that its recurrence can be parameterized by an associative
binary operation, enabling efficient training and inference over long sequence by an
[parallel associate-scan](https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithms)
operation. Here's what it looks like to translate the 

Credit to 
"S5: Simplified State Space Layers for Sequence Modeling" (https://arxiv.org/abs/2208.04933)
Credit to https://github.com/lindermanlab/S5


```python
def ssm(
    x:  Float[Array, "L D  "],
    dt: Float[Array, "L D  "],
    A:  Float[Array, "  D N"],
    B:  Float[Array, "L   N"],
    C:  Float[Array, "L   N"],
    D:  Float[Array, "  D  "],
) -> Float[Array, "L D"]:
    """Selective Scan operation."""
    l, d = x.shape
 
    # Discretize the continuous-time SSM (implementation omitted). See paper.
    dA, dB = discretize(A, B, dt)

    def binop(
        ei: tuple[Float[Array, "d n"], Float[Array, "d n"]],
        ej: tuple[Float[Array, "d n"], Float[Array, "d n"]],
    ):
        """See appendix H of S4 paper for detailed review of associate scan for LTI."""
        Ai, Bxi = ei
        Aj, Bxj = ej
        return Ai * Aj, Aj * Bxi + Bxj

    dBx: Float[Array, "l d n"] = dB * x[:, :, None]
    A_, B_ = jax.lax.associative_scan(binop, (dA, dBx))

    h: Float[Array, "l d n"] = A_ + B_
    y = einops.einsum(h, C, "l d n, l n -> l d")
    return y + x * D
```

Care must be taken to retain the associativity of the binary operation when considering 
how to integrate boupndary resetting. With a naive implementaiton, we'll end up
violating the 

Consider this pseudo-code where we (naively) try to implement reset gates

```python
def f(
    ei: tuple[Float[Array, "n"], Float[Array, "n"], Bool],
    ej: tuple[Float[Array, "n"], Float[Array, "n"], Bool],
):
    """Naive Implementation! Breaks associativity!"""
    Ai, Bi, reset_i = ei
    Aj, Bj, reset_j = ej
    if reset_i:
        return Aj, Bj, rj
    if reset_j:
        return zeros_like(Aj), zeros_like(Bj), rj
    else:
        return Ai * Aj, Aj * Bi + Bj
```
To understand why this won't work, consider this example of scanning the sequence
`[X, r, Y]` where `X` and `Y` are inputs and `r` is a boundary-reset indicator.

If we associate on the left, we arrive at result `Y`:
`X, r, Y -> f(X, r), Y -> r, Y -> f(r, Y) -> Y`

However, if we associate on the right, we'll get `f(X, Y)`
`X, r, Y -> X, f(r, Y) -> X, Y -> f(X, Y)`

The inequality of these two results demonstrates naive `f`'s non-associativity.

### Associativity Fix

The key to fixing associativity is thinking about the operation like a cumulative sum-
the simplest operation that can be made efficient with an associatve scan.

If we keep track of the *cumulative total number of reset boundaries* that have been
encountered so far, we can decide if the two states should be merged. 

Again, in pseudo-code:

```python


def f(
    ei: tuple[Float[Array, "n"], Float[Array, "n"], Bool, Integer],
    ej: tuple[Float[Array, "n"], Float[Array, "n"], Bool, Integer],
):
    Aj, Bj, rj, count_j = ej
    Ai, Bi, ri, count_i = ei

    if ri:
        # Only increment when reset comes from left.
        return Aj, Bj, rj, count_j + 1

    if rj:
        return jnp.zeros_like(Aj), jnp.zeros_like(Bj), rj, count_j

    if count_i < count_j:
        return Aj, Bj, rj, count_j

    A = Ai * Aj
    B = Aj * Bi + Bj
    return A, B, rj, jnp.maximum(count_i, count_j)
```

