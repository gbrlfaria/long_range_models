from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .types import Array, SequenceLayer


def bidirectional(layer: SequenceLayer, project_outputs: bool = False) -> SequenceLayer:
    class Bidirectional(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, x: Array, *args, **kwargs) -> Array:
            fwd = layer(self.dim)
            bwd = layer(self.dim)

            x_fwd = fwd(x, *args, **kwargs)
            x_bwd = bwd(x[:, ::-1, ...], *args, **kwargs)[:, ::-1, ...]
            x = jnp.concatenate((x_fwd, x_bwd), axis=2)

            if project_outputs:
                x = nn.Dense(self.dim)(x)

            return x

    return Bidirectional


def activation(name: str) -> Callable[[Array], Array]:
    if name == "id":
        return lambda x: x
    if name == "relu":
        return nn.relu
    if name == "leaky_relu":
        return nn.leaky_relu
    if name == "sigmoid":
        return nn.sigmoid
    if name == "tanh":
        return nn.tanh
    if name == "gelu":
        return nn.gelu
    if name in ["swish", "silu"]:
        return nn.swish
    if name == "celu":
        return nn.celu
    if name == "elu":
        return nn.elu
    if name == "hard_tanh":
        return nn.hard_tanh
    if name == "hard_sigmoid":
        return nn.hard_sigmoid
    if name in ["hard_swish", "hard_silu"]:
        return nn.hard_swish
    if name == "glu":
        return nn.glu
    if name == "half_glu":
        return nn.glu

    raise ValueError(f"Unknown activation function '{name}'")


def dense_activation(dim: int, activation_name: str) -> Callable[[Array], Array]:
    # Create dense layer
    if activation_name == "glu":
        dense = nn.Dense(dim * 2)
    elif activation_name == "half_glu":
        # Only linear on the gating side of GLU
        half_dense = nn.Dense(dim)
        dense = lambda x: jnp.concatenate([x, half_dense(x)], axis=-1)
    else:
        dense = nn.Dense(dim)

    # Create activation function
    act = activation(activation_name)

    return lambda x: act(dense(x))
