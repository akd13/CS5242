import numpy as np
from itertools import product


def clip_gradients(in_grads, clip=1):
    return np.clip(in_grads, -clip, clip)


def rel_error(x, y):
    return np.mean(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def img2col(data, h_indices, w_indices, k_h, k_w):
    batch = data.shape[0]
    indices = list(product(h_indices, w_indices))
    out = np.stack(map(
        lambda x: data[:, :, x[0]:x[0]+k_h, x[1]:x[1]+k_w].reshape(batch, -1), indices), axis=-1)
    return out

def col2img(cur_in_grad, col2img_input,out_height,out_width,in_channel,kernel_h,kernel_w,stride):
    # refer: https://github.com/fanghao6666/neural-networks-and-deep-learning/
    for h in range(out_height):
        for w in range(out_width):
            for c in range(in_channel):
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        cur_in_grad[c,h*stride+kh,w*stride+kw]+=col2img_input[c*kernel_h*kernel_w+kh*kernel_w+kw,h*out_width+w]
    return cur_in_grad