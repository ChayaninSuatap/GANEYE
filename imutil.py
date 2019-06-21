from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

GAUSSIAN = 0
MEDIAN = 1

def gaussian_blur(im, blur_val):
    im = Image.fromarray(im.astype('uint8'), 'RGB').filter(ImageFilter.GaussianBlur(blur_val))
    return np.array(im)

def median_filter(im, blur_val):
    im = Image.fromarray(im.astype('uint8'), 'RGB').filter(ImageFilter.MedianFilter(blur_val))
    return np.array(im)

def make_edge(im, blur_val, blur_fn):
    if blur_fn == GAUSSIAN:
        blured = gaussian_blur( im, blur_val)
    elif blur_fn == MEDIAN:
        blured = median_filter( im, blur_val)
    else:
        input('error')
    blured = blured.astype('float32')
    im = im.astype('float32')
    edge = im - blured
    return edge