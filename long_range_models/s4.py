"""This module implements the models described in the paper [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)"""

from typing import Callable, Optional

import flax.linen as nn

from .ops import Bidirectional, HalfGLU1
from .types import Activation, Array, SequenceLayer


class S4Module(nn.Module):
    """Implementation of the stacked architecture module proposed in the S4 paper. \
    Default parameters are based on S5's interpretation of this architecture.

    Attributes:
        dim: \
            The dimension of the input and output embeddings/features.
        depth: \
            The number of blocks/layers within the model.
        sequence_layer: \
            The type of sequence layer used within the model.
        activation: \
            pass
        gate: \
            pass
        bidirectional: \
            pass
    """

    dim: int
    depth: int
    sequence_layer: SequenceLayer
    activation: Optional[Activation] = nn.gelu
    gate: Optional[Callable[[int, Activation], Callable[[Array], Array]]] = HalfGLU1
    bidirectional: bool = False
    skip: bool = True
    norm: Optional[str] = "batch"
    prenorm: bool = True
    dropout: float = 0.0
    tie_dropout: bool = True

    @nn.compact
    def __call__(self, inputs: Array, train: bool) -> Array:
        """Applies the forward pass of `S4Module` to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, dim)`.

        Returns:
            The output data with shape `(batch_size, input_length, dim)`.
        """

        # Initialize sequence layer
        if not self.bidirectional:
            sequence_layer = self.sequence_layer(self.dim)
        else:
            sequence_layer = nn.Sequential(
                [
                    Bidirectional(
                        forward_module=self.sequence_layer(self.dim),
                        backward_module=self.sequence_layer(self.dim),
                    ),
                    nn.Dense(2 * self.dim, self.dim),
                ]
            )

        # Initialize normalization layer
        if self.norm == "batch":
            norm = nn.BatchNorm(use_running_average=not train)
        elif self.norm == "layer":
            norm = nn.LayerNorm()
        elif self.norm is None:
            norm = lambda x: x
        else:
            raise ValueError(f"Invalid normalization type `{self.norm}`")

        # Initialize dropout layer
        drop = nn.Dropout(
            rate=self.dropout,
            broadcast_dims=[1] if self.tie_dropout else (),
            deterministic=not train,
        )

        # Set activation function + dropout
        if self.activation is not None:
            activation = lambda x: drop(self.activation(x))
        else:
            activation = lambda x: drop(x)

        # Forward pass
        x = inputs
        for _ in range(self.depth):
            skip = x

            # Pre-normalization if enabled
            if self.prenorm:
                x = norm(x)

            # Apply sequence layer
            x = sequence_layer(x)

            # Post layer operation (activation or gate)
            if self.gate is not None:
                x = drop(self.gate(self.dim, activation)(x))
            else:
                x = activation(x)

            # Residual connection
            if self.skip:
                x = x + skip

            # Post-normalization if not pre-normalized
            if not self.prenorm:
                x = norm(x)

        return x
