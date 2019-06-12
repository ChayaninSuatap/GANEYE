from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pix2pix_eye import Pix2Pix
import os
import PIL

print('loading model')
o = Pix2Pix(gen_weights_fn='gen_ep-8000-sample-0.hdf5', dis_weights_fn='dis_ep-8000-sample-0.hdf5', load_for_predict=True, save_path='saved_model_eyes')
model = o.combined
print('loaded')

#predict 
folder_path = 'my_blues'
for fn in os.listdir('my_blues'):
    path = 'my_blues/' + fn 
    img = scipy.misc.imread(path, mode='RGB').astype(np.float)
    img = scipy.misc.imresize(img, (256,256))
    xs = np.array([img])/127.5 - 1.
    xs = o.make_imgb_with_label(xs,[0 if fn[0]=='0' else 1])

    print(fn)
    pred = model.predict( xs)
    # plt.imshow(((pred[1][0] * 0.5 + 0.5) * 256))
    # plt.show()
    scipy.misc.imsave('my_blues_result/' + fn[:-3] + 'png', (pred[1][0] * 0.5 + 0.5))
    # plt.savefig('my_blues_result/' + fn)


