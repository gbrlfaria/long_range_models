from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import metrics
from flax import struct
from flax.training import train_state

from long_range_models import ContinuousSequenceModel, LRULayer, S4Module, S5Layer
from long_range_models.types import Array


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def get_datasets(num_epochs: int, batch_size: int):
    train_ds = tfds.load("cifar10", split="train", as_supervised=True)
    test_ds = tfds.load("cifar10", split="test", as_supervised=True)

    # Define transform function
    def transform_fn(image, label) -> Tuple[Array, Array]:
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)
        image = (image / 255.0 - 0.5) * 2.0
        return tf.reshape(image, (-1, image.shape[-1])), label

    train_ds = train_ds.map(transform_fn)
    test_ds = test_ds.map(transform_fn)

    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.shuffle(1024)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds


# Initialize seed
rng_init = jrandom.PRNGKey(1919)

model = ContinuousSequenceModel(
    module=S4Module(
        sequence_layer=partial(S5Layer, state_dim=256, num_blocks=8, conj_sym=True),
        dim=128,
        depth=6,
    ),
    out_dim=10,
    pool=True,
)

# Initialize parameters
dummy_batch = jnp.ones((1, 32 * 32, 3))
params = model.init(rng_init, dummy_batch)

# Initialize optimizer
tx = optax.adamw(learning_rate=1e-3, weight_decay=0.05)

# Get train state
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    metrics=Metrics.empty(),
)


# Define loss function
def loss_fn(output: Array, target: Array) -> Array:
    return optax.softmax_cross_entropy_with_integer_labels(output, target).mean()


@jax.jit
def train_step(state: TrainState, batch: Tuple[Array, Array]) -> TrainState:
    inputs, targets = batch

    def apply_loss(params: Dict) -> Array:
        outputs = state.apply_fn(params, inputs)
        return loss_fn(outputs, targets)

    grad_fn = jax.grad(apply_loss)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@jax.jit
def compute_metrics(
    state: TrainState,
    batch: Tuple[Array, Array],
) -> TrainState:
    inputs, targets = batch
    outputs = state.apply_fn(state.params, inputs)
    loss = loss_fn(outputs, targets)
    metric_updates = state.metrics.single_from_model_output(
        logits=outputs, labels=targets, loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


num_epochs = 100
batch_size = 32
train_ds, test_ds = get_datasets(num_epochs, batch_size)

num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
metrics_history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    state = train_step(state, batch)
    state = compute_metrics(state, batch)

    if (step + 1) % num_steps_per_epoch == 0:
        for metric, value in state.metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)

        # Reset metrics
        state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on the test set after each training epoch
        test_state = state
        for test_batch in test_ds.as_numpy_iterator():
            test_state = compute_metrics(state=test_state, batch=test_batch)

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)

        print(
            f"train epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['train_loss'][-1]}, "
            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}"
        )
        print(
            f"test epoch: {(step+1) // num_steps_per_epoch}, "
            f"loss: {metrics_history['test_loss'][-1]}, "
            f"accuracy: {metrics_history['test_accuracy'][-1] * 100}"
        )
