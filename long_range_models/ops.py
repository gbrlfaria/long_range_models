import flax.linen as nn
import jax.numpy as jnp

from .types import Activation, Array


class GLU(nn.Module):
    dim: int
    act: Activation

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.act(x)
        return nn.Dense(self.dim)(x) * nn.sigmoid(nn.Dense(self.dim)(x))


class HalfGLU1(nn.Module):
    dim: int
    act: Activation

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.act(x)
        return x * nn.sigmoid(nn.Dense(self.dim)(x))


class HalfGLU2(nn.Module):
    dim: int
    act: Activation

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x * nn.sigmoid(nn.Dense(self.dim)(self.act(x)))


class Bidirectional(nn.Module):
    forward_module: nn.Module
    backward_module: nn.Module

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        x_fwd = self.forward_module(x, *args, **kwargs)
        x_bwd = self.backward_module(x[:, ::-1, ...], *args, **kwargs)[:, ::-1, ...]
        return jnp.concatenate((x_fwd, x_bwd), axis=2)
