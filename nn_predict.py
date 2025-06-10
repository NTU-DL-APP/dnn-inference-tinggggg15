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
    x = np.asarray(x, dtype=np.float64)                  # 確保是浮點數
    # 1) 在最後一個維度上減去最大值以提升穩定性
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    # 2) 計算指數
    exps = np.exp(x_shifted)
    # 3) 在最後一個維度上求和，然後逐元素相除
    sums = np.sum(exps, axis=-1, keepdims=True)
    return exps / sums

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    # 1. 把 arr_0,arr_1,… 按數字排好
    x = data
    idx = 0
    for layer in model_arch:
        ltype = layer['type']
        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W, b = weights[idx], weights[idx+1]
            idx += 2
            x = dense(x, W, b)
            act = layer['config'].get("activation", "")
            if act == "relu":
                x = relu(x)
            elif act == "softmax":
                x = softmax(x)
        else:
            raise ValueError(f"Unsupported layer type: {ltype}")
    return x
    # x = data
    # for layer in model_arch:
    #     lname = layer['name']
    #     ltype = layer['type']
    #     cfg = layer['config']
    #     wnames = layer['weights']

    #     if ltype == "Flatten":
    #         x = flatten(x)
    #     elif ltype == "Dense":
    #         W = weights[wnames[0]]
    #         b = weights[wnames[1]]
    #         x = dense(x, W, b)
    #         if cfg.get("activation") == "relu":
    #             x = relu(x)
    #         elif cfg.get("activation") == "softmax":
    #             x = softmax(x)

    # return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)
    
