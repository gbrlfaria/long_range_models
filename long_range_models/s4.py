"""This module implements the models described in the paper [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)"""

import flax.linen as nn

from .types import Array, SequenceLayer


class S4Module(nn.Module):
    """Implementation of the model architecture used in the S4 paper.

    Attributes:
        dim: \
            The dimension of the input and output embeddings.
        depth: \
            The number of blocks/layers within the model.
        sequence_layer: \
            The type of sequence layer used within the model.
    """

    dim: int
    depth: int
    sequence_layer: SequenceLayer

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies the forward pass of `S4Module` to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, dim)`.

        Returns:
            The output data with shape `(batch_size, input_length, dim)`.
        """

        x = inputs
        for _ in range(self.depth):
            skip = x
            x = nn.LayerNorm()(x)
            x = self.sequence_layer(self.dim)(x)
            x = nn.gelu(x) * nn.sigmoid(nn.Dense(self.dim)(nn.gelu(x)))
            x = x + skip
        return x
