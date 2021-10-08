import jax.numpy as jnp
from jax.experimental.ode import odeint
import numpy.random as npr
from jax import jit, grad


def relu(x):
    return jnp.maximum(0.0, x)

def mlp(params, inputs):
# A multi-layer perceptron, i.e. a fully-connected neural network.
    for w, b in params:
        outputs = jnp.dot(inputs, w) + b  # Linear transform
        inputs = jnp.tanh(outputs)            # Nonlinearity
    return outputs

def nn_dynamics(state, time, params):
    state_and_time = jnp.hstack([state, jnp.array(time)])
    return mlp(params, state_and_time)


def odenet(params, input):
    start_and_end_times = jnp.array([0.0, 1.0])
    init_state, final_state = odeint(nn_dynamics, input, start_and_end_times, params, atol=0.001, rtol=0.001)
    return final_state

def odenet_loss(params, model, inputs, targets):
    preds = model(params, inputs)
    return jnp.mean(jnp.sum((preds - targets)**2, axis=1))

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def odenet_update(params, step_size, inputs, targets):
    grads = grad(odenet_loss)(params, inputs, targets)
    return [(w - step_size * dw, b - step_size * db)
                  for (w, b), (dw, db) in zip(params, grads)]

