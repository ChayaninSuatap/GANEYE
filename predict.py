from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pix2pix_no_blue import Pix2Pix

print('loading model')
o = Pix2Pix(15, 'no_blue_gen_ep-15-sample-200.hdf5', 'no_blue_dis_ep-15-sample-200.hdf5')
model = o.combined
print('loaded')
img = scipy.misc.imread('my_input_2.jpg', mode='RGB').astype(np.float)
img = scipy.misc.imresize(img, (256,256))
xs = np.array([img])/127.5 - 1.
print('xs shape', xs.shape)
pred = model.predict( xs)
# print(np.array(pred).shape)
print(pred)
plt.imshow(pred[1][0] * 0.5 + 0.5)
plt.show()


