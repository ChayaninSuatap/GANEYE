from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(im, blur_val=3):
    im = Image.fromarray(im.astype('uint8'), 'RGB').filter(ImageFilter.GaussianBlur(blur_val))
    print('this is guassian')
    return np.array(im)

def median_filter(im, blur_val=31):
    im = Image.fromarray(im.astype('uint8'), 'RGB').filter(ImageFilter.MedianFilter(blur_val))
    return np.array(im)

def make_edge(im, blur_val=3, blur_fn=gaussian_blur):
    blured = blur_fn( im, blur_val)
    blured = blured.astype('float32')
    im = im.astype('float32')
    edge = im - blured
    return edge