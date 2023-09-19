# Long Range Models

[![PyPI version](https://badge.fury.io/py/long-range-models.svg)](https://badge.fury.io/py/long-range-models)
[![Static Badge](https://img.shields.io/badge/written%20in-Flax-blue)](https://github.com/google/flax)
[![Static Badge](https://img.shields.io/badge/license-MIT-yellow)](/LICENSE)

A collection of simple implementations of long-range sequence models and layers, including [S4](/long_range_models/s4.py), [S5](/long_range_models/s5.py), and the [LRU](/long_range_models/lru.py).
More implementations to come.

## Install

```bash
$ pip install long-range-models
```

## Usage

This library offers detailed documentation for every module implemented.
Models are created by integrating different components (layers, backbones, and sequence model wrappers).
See some minimal examples below.

### Discrete sequence data

Consider a language model built with an LRU sequence layer and an S4 backbone:

```py
import jax.random as jrandom
from functools import partial
from long_range_models import SequenceModel, S4Backbone, LRULayer

rng = jrandom.PRNGKey(0)

model = SequenceModel(
  num_tokens=1000,
  module=S4Backbone(
    sequence_layer=partial(LRULayer, state_dim=256),
    dim=128,
    depth=6,
  ),
)

x = jrandom.randint(rng, (1, 1024), 0, 1000)

variables = model.init(rng, x, train=False)
model.apply(variables, x, train=False)  # (1, 1024, 1000)

```

### Continuous sequence data

For continuous-valued sequences, the setup looks as follows:

```py
import jax.random as jrandom
from functools import partial
from long_range_models import ContinuousSequenceModel, S4Backbone, LRULayer

rng = jrandom.PRNGKey(0)

model = ContinuousSequenceModel(
  out_dim=10,
  module=S4Backbone(
    sequence_layer=partial(LRULayer, state_dim=256),
    dim=128,
    depth=6,
  ),
)

x = jrandom.normal(rng, (1, 1024, 32))

variables = model.init(rng, x, train=False)
model.apply(variables, x, train=False)  # (1, 1024, 10)

```

Both model types offer several customization options.
Make sure to check out [their documentation](/long_range_models/sequence_models.py).
Also check out full examples [here](/examples/).

### Bidirectional sequence layers

You can create bidirectional sequence layers by using the `bidirectional` wrapper function:

```py
from long_range_models import bidirectional

sequence_layer = bidirectional(partial(LRULayer, state_dim=256))
```

## Upcoming features

- **More implementations:** extend the library with models like S4D, S4Liquid, BiGS, Hyena, RetNet, SGConv, H3, and others.
- **Customization:** allow users to better customize currently implemented layers and architectures (e.g., activation functions, initialization, etc.).
- **Sequential API:** allow recurrent models to run sequentially, allowing for efficient inference.
