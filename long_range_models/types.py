"""This module defines generic types used as annotations throughout the library."""

from typing import Any, Callable

import jax

Array = jax.Array
PRNGKey = Any
Shape = Any
DType = Any

Initializer = Callable[[PRNGKey, Shape, DType], Array]

SequenceLayer = Callable[[int], Callable[[Array], Array]]
