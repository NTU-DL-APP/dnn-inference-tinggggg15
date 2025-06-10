import numpy as np
import json

# === Activation functions ===
def relu(x):
    # TODO: Implement the Rectified Linear Unit
    # return x
    return np.maximum(0, x)

def softmax(x):
    # TODO: Implement the SoftMax function
    # return x
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x_shifted)
    sum_exps = np.sum(exps, axis=-1, keepdims=True)
    return exps / sum_exps

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    
