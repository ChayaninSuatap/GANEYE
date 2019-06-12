from PIL import Image, ImageFilter
import numpy as np

def gaussian_blur(im, blur_val=3):
    im = Image.fromarray(im.astype('uint8'), 'RGB').filter(ImageFilter.GaussianBlur(blur_val))
    return np.array(im)

def make_edge(im, blur_val=3):
    blured = gaussian_blur( im, blur_val)
    edge = im - blured
    edge = edge.clip(0, 255)
    return edge