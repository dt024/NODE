import matplotlib.pyplot as plt
import numpy as np

def draw_graph(inputs, targets, odenet_params, model):
    # Plot resulting model.
    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.gca()
    ax.scatter(inputs, targets, lw=0.5, color='green')
    # fine_inputs = jnp.reshape(scaler.transform(np.array(train_data.time_stamps[0:10]).reshape(-1,1)), (10, 1))
    # fine_inputs = jnp.reshape(jnp.linspace(-3.0, 3.0, 10), (10, 1))
    ax.plot(input, model(odenet_params, inputs), lw=0.5, color='red')
    ax.set_xlabel('input')
    ax.set_ylabel('output')
    plt.legend(("ODE Net predictions","Ground Truth"))