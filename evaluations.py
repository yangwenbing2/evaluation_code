"""
Reimplement evaluation.mat provided by Adobe in python
Output of `compute_gradient_loss` is sightly different from the MATLAB version provided by Adobe (less than 0.1%)
Output of `compute_connectivity_error` is smaller than the MATLAB version (~5%, maybe MATLAB has a different algorithm)
So do not report results calculated by these functions in your paper.
Evaluate your inference with the MATLAB file `DIM_evaluation_code/evaluate.m`.

by Yaoyi Li
"""
"""
Modified by Wenbing Yang
2022.09.01
"""

import scipy.ndimage
import numpy as np
from skimage.measure import label
import scipy.ndimage.morphology


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient_loss(pred, target, trimap=None):
    pred = pred
    target = target

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    if trimap is not None:
        loss = np.sum(error_map[trimap == 128])
    else:
        loss = np.sum(error_map)
    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap=None, step=0.1):
    pred = pred
    target = target
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)

    if trimap is not None:
        loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])
    else:
        loss = np.sum(np.abs(pred_phi - target_phi))

    return loss / 1000.


def compute_mse_loss(pred, target, trimap=None):
    w, h = pred.shape[:2]
    error_map = (pred - target) ** 2

    if trimap is not None:
        loss = np.sum((error_map) * (trimap == 1))
        pixel = np.sum(trimap == 1) + 1e-8
    else:
        loss = np.sum((error_map))
        pixel = w * h + 1e-8

    return loss / pixel


def compute_sad_loss(pred, target, trimap=None):
    error_map = np.abs(pred - target)

    if trimap is not None:
        loss = np.sum(error_map * (trimap == 1))
    else:
        loss = np.sum(error_map)

    return loss / 1000


def compute_mad_loss(pred, target, trimap=None):
    w, h = pred.shape[:2]
    error_map = np.abs(pred - target)

    if trimap is not None:
        loss = np.sum(error_map * (trimap == 1))
        pixel = np.sum(trimap == 1) + 1e-8
    else:
        loss = np.sum(error_map)
        pixel = w * h + 1e-8

    return loss / pixel


# Creared by
# 2022.09.01
def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0] * predict.shape[1]

    sad_diff = np.sum(np.abs(predict - alpha)) / 1000
    mse_diff = np.sum((predict - alpha) ** 2) / pixel
    mad_diff = np.sum(np.abs(predict - alpha)) / pixel

    return sad_diff, mse_diff, mad_diff
