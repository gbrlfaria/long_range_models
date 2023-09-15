"""This module provides sequence modeling wrappers."""

from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp

from .types import Array


class SequenceModel(nn.Module):
    """A sequence model wrapper for discrete token sequences.
    
    This model wraps around a given sequence processing module. It first transforms \
    input tokens using an embedding layer, passes them through the inner module, and \
    finally decodes them through a dense layer.

    Attributes:
        module: \
            The inner sequence processing module.
        num_tokens: \
            The number of unique input tokens.
        emb_dim: \
            The embedding/feature dimension. Defaults to the `dim` attribute \
            of the inner module if not specified.
        out_dim: \
            The dimension of the output of the model. Defaults to `num_tokens` if \
            not specified.
        pool: \
            If `True`, applies mean pooling over the sequence dimension before the \
            output layer.
    """

    module: nn.Module
    num_tokens: int
    emb_dim: Optional[int] = None
    out_dim: Optional[int] = None
    pool: bool = False

    @nn.compact
    def __call__(self, inputs: Array, *args: Any, **kwargs: Any) -> Array:
        """Applies the sequence model to the input tokens.

        Args:
            inputs: \
                The input array of token IDs with shape `(batch_size, input_length)`.
            *args: \
                Additional arguments to be passed to the inner module call.
            **kwargs: \
                Additional keyword arguments to be passed to the inner module call.

        Returns:
            The output data with shape `(batch_size, input_length, out_dim)`.
        """

        # Variable aliasing (for convenience)
        module = self.module
        num_tokens = self.num_tokens
        emb_dim = self.emb_dim
        out_dim = self.out_dim
        pool = self.pool

        # Apply defaults
        out_dim = out_dim if (out_dim is not None) else num_tokens
        emb_dim = emb_dim if (emb_dim is not None) else module.dim

        # Forward pass
        return sequence_model(
            module=module,
            embed=nn.Embed(num_tokens, emb_dim),
            out_dim=out_dim,
            pool=pool,
        )(inputs, *args, **kwargs)


class ContinuousSequenceModel(nn.Module):
    """A sequence model wrapper for continuous-valued sequences.
    
    This model wraps around a given sequence processing module. It first transforms \
    input data using a dense layer, passes them through the inner module, and finally \
    decodes them with another dense layer.

    Attributes:
        module: \
            The inner sequence processing module.
        out_dim: \
            The dimension of the output of the model.
        emb_dim: \
            The embedding/feature dimension. Defaults to the `dim` attribute \
            of the inner module if not specified.
        pool: \
            If `True`, applies mean pooling over the sequence dimension before the \
            output layer.
    """

    module: nn.Module
    out_dim: int
    emb_dim: Optional[int] = None
    pool: bool = False

    @nn.compact
    def __call__(self, inputs: Array, *args: Any, **kwargs: Any) -> Array:
        """Applies the sequence model to the input data.

        Args:
            inputs: \
                The input data with shape `(batch_size, input_length, input_dim)`.
            *args: \
                Additional arguments to be passed to the inner module call.
            **kwargs: \
                Additional keyword arguments to be passed to the inner module call.

        Returns:
            The output data with shape `(batch_size, input_length, out_dim)`.
        """

        # Variable aliasing (for convenience)
        module = self.module
        out_dim = self.out_dim
        emb_dim = self.emb_dim
        pool = self.pool

        # Apply defaults
        emb_dim = emb_dim if (emb_dim is not None) else module.dim

        # Forward pass
        return sequence_model(
            module=module,
            embed=nn.Dense(emb_dim),
            out_dim=out_dim,
            pool=pool,
        )(inputs, *args, **kwargs)


def sequence_model(module: nn.Module, embed: nn.Module, out_dim: int, pool: bool):
    def forward(inputs: Array, *args: Any, **kwargs: Any) -> Array:
        x = embed(inputs)
        x = module(x, *args, **kwargs)
        if pool:
            x = jnp.mean(x, axis=1)
        x = nn.Dense(out_dim)(x)
        return x

    return forward
