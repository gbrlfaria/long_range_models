"""This module implements the models described in the paper [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)"""

from typing import Optional

import flax.linen as nn

from .ops import activation, dense_activation
from .types import Array, SequenceLayer


class S4Backbone(nn.Module):
    """Implementation of the sequence model backbone proposed in the S4 paper.

    Attributes:
        dim: \
            The dimension of the input and output embeddings/features.
        depth: \
            The number of blocks/layers within the model.
        sequence_layer: \
            The type of sequence layer used within the model.
        act: \
            The activation function to be applied immediately after the sequence \
            layer. If `None`, no activation is used. Refer to the `ops` module \
            for available options.
        ffn_act: \
            The activation function to be applied after the feedforward layer that \
            follows the sequence layer. If `None` no activation nor feedforward layer \
            is used. Refer to the `ops` module for available options.
        skip: \
            Whether skip/residual connections should be used in each block/layer.
        norm: \
            The type of normalization to be applied. Options include `"batch"` for \
            batch normalization, `"layer"` for layer normalization, and `None` \
            for no normalization.
        prenorm: \
            Whether normalization should be applied before the sequence layer. \
            If `False`, normalization is applied after the skip connection.
        dropout: \
            The dropout probability.
        tie_dropout: \
            Whether the dropout mask should be shared across the length of the sequence.
    """

    dim: int
    depth: int
    sequence_layer: SequenceLayer
    act: Optional[str] = "gelu"
    ffn_act: Optional[str] = "glu"
    skip: bool = True
    norm: Optional[str] = "batch"
    prenorm: bool = True
    dropout: float = 0.0
    tie_dropout: bool = True

    @nn.compact
    def __call__(self, inputs: Array, train: bool) -> Array:
        """Applies the `S4Backbone` module to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, dim)`.
            train: \
                Whether the model is being trained. It is used for correctly \
                configuring the dropout and normalization layers.

        Returns:
            The output data with shape `(batch_size, input_length, dim)`.
        """

        x = inputs
        for _ in range(self.depth):
            x = S4Block(
                self.dim,
                self.sequence_layer,
                self.act,
                self.ffn_act,
                self.skip,
                self.norm,
                self.prenorm,
                self.dropout,
                self.tie_dropout,
            )(x, train)
        return x


class S4Block(nn.Module):
    """A block of the `S4Backbone` module."""

    dim: int
    sequence_layer: SequenceLayer
    act: Optional[str]
    ffn_act: Optional[str]
    skip: bool
    norm: Optional[str]
    prenorm: bool
    dropout: float
    tie_dropout: bool

    @nn.compact
    # TODO: pass args and kwargs into sequence layers
    def __call__(self, x: Array, train: bool) -> Array:
        # Initialize sequence layer
        sequence_layer = self.sequence_layer(self.dim)

        # Initialize normalization layer
        if self.norm == "batch":
            norm = nn.BatchNorm(use_running_average=not train)
        elif self.norm == "layer":
            norm = nn.LayerNorm()
        elif self.norm is None:
            norm = lambda x: x
        else:
            raise ValueError(f"Unknown normalization type '{self.norm}'")

        # Initialize dropout layer
        drop = nn.Dropout(
            rate=self.dropout,
            broadcast_dims=[1] if self.tie_dropout else (),
            deterministic=not train,
        )

        # Set activation function + dropout
        if self.act is not None:
            act = lambda x: drop(activation(self.act)(x))
        else:
            act = lambda x: drop(x)

        # Set FFN + activation + dropout
        if self.ffn_act is not None:
            ffn_act = lambda x: drop(dense_activation(self.dim, self.ffn_act)(x))
        else:
            ffn_act = lambda x: x

        # Forward pass
        skip = x

        # Pre-normalization if enabled
        if self.prenorm:
            x = norm(x)

        # Apply sequence layer and activation function
        x = sequence_layer(x)
        x = act(x)

        # Post layer operation
        x = ffn_act(x)

        # Residual connection
        if self.skip:
            x = x + skip

        # Post-normalization if not pre-normalized
        if not self.prenorm:
            x = norm(x)

        return x
