"""This module defines generic types used as annotations throughout the library."""

from typing import Callable

import jax

Array = jax.Array

SequenceLayer = Callable[[int], Callable[[Array], Array]]
