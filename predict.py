from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pix2pix_eye import Pix2Pix
import os
import PIL
import imutil

print('loading model')
o = Pix2Pix(gen_weights_fn='gen_ep-8000-sample-0.hdf5', dis_weights_fn='dis_ep-8000-sample-0.hdf5', load_for_predict=True, save_path='saved_model_eyes')
o_edge = Pix2Pix(gen_weights_fn='gen.hdf5', dis_weights_fn='dis.hdf5', load_for_predict=True, save_path='saved_model_eyes')
model = o.combined
model_edge = o_edge.combined
print('loaded')

#predict 
folder_path = 'my_blues'
for fn in os.listdir('my_blues'):
    path = 'my_blues/' + fn 
    img = scipy.misc.imread(path, mode='RGB').astype(np.float)
    img = scipy.misc.imresize(img, (256,256))
    xs = np.array([img])/127.5 - 1.
    xs = o.make_imgb_with_label(xs, [0 if fn[0]=='0' else 1])

    print(fn)
    pred = model.predict( xs)
    pred_edge = model_edge.predict( xs)
    im = ((pred[1][0] * 0.5 + 0.5)).astype('float32')
    im_edge = ((pred_edge[1][0] / 255)).astype('float32')
    im_final = im_edge + im
    
    print(im_edge)
    grid = plt.GridSpec(1,3)
    plt_im = plt.subplot(grid[0,0])
    plt_edge = plt.subplot(grid[0,1])
    plt_final = plt.subplot(grid[0,2])
    plt_im.imshow(im)
    plt_edge.imshow(im_edge)
    plt_final.imshow(im_final)
    plt.show()
    # scipy.misc.imsave('my_blues_result/' + fn[:-3] + 'png', im)
    # plt.savefig('my_blues_result/' + fn)


