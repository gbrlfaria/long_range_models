from typing import Callable

import flax.linen as nn
import jax.numpy as jnp

from .types import Array, SequenceLayer


def bidirectional(layer: SequenceLayer, project_outputs: bool = False) -> SequenceLayer:
    """Creates a bidirectional sequence layer from a unidirectional sequence layer.

    The layer works by running one instance of the sequence layer in each direction \
    and then concatenating their outputs in the feature axis, doubling the feature \
    dimension. Optionally, the concatenated outputs can be linearly projected to match \
    the feature dimension of the inputs.

    Args:
        layer: \
            The unidirectional sequence layer.
        project_outputs: \
            Whether to project the outputs of the bidirectional layer to match the \
            feature dimension of the inputs.

    Returns:
        The corresponding bidirectional sequence layer.
    """

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
    """Returns the desired activation function based on the given name."""

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
    """
    Returns a composed function of a dense layer followed by the desired activation.
    """

    # Create dense layer
    if activation_name == "glu":
        dense = nn.Dense(dim * 2)
    elif activation_name == "half_glu":
        # In the half GLU, only the gate input goes through the feedforward layer
        gate_dense = nn.Dense(dim)
        dense = lambda x: jnp.concatenate([x, gate_dense(x)], axis=-1)
    else:
        dense = nn.Dense(dim)

    # Create activation function
    act = activation(activation_name)

    return lambda x: act(dense(x))
