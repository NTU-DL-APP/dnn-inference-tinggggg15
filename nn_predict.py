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
def nn_forward_h5(model_arch, raw_weights, data):
    # 1. 把 arr_0,arr_1,… 按數字排好
    # 1) 如果是 NpzFile，就按 layer/bias 顺序把它转成 list
    if hasattr(raw_weights, 'files'):
        npz = raw_weights
        def sort_key(name):
            # 'dense1_kernel' → 层号 1，kernel 排 bias 之前
            idx = int(name.replace('dense', '').split('_')[0])
            rank = 0 if 'kernel' in name else 1
            return (idx, rank)
        keys = sorted(npz.files, key=sort_key)
        weight_list = [npz[k] for k in keys]
    else:
        weight_list = raw_weights

    # 2) 如果 model_arch 是 dict（Keras JSON），抽出它的 layers list
    if isinstance(model_arch, dict) and 'config' in model_arch:
        layers = model_arch['config']['layers']
    else:
        layers = model_arch

    # 3) 依序做 forward
    x = data
    w_idx = 0
    for layer in layers:
        # class_name for JSON, or type if you passed a simpler list
        ltype = layer.get('class_name', layer.get('type'))
        cfg   = layer['config']

        if ltype == 'Flatten':
            x = flatten(x)

        elif ltype == 'Dense':
            W, b = weight_list[w_idx], weight_list[w_idx+1]
            w_idx += 2
            x = dense(x, W, b)

            act = cfg.get('activation', '')
            if act == 'relu':
                x = relu(x)
            elif act == 'softmax':
                x = softmax(x)

        else:
            # 如果有别的层，可以在这儿 extend
            continue

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
    
