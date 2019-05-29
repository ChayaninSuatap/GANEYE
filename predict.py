from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pix2pix_no_blue import Pix2Pix

print('loading model')
model = load_model('saved_model/model_no_blue_ep-4-sample_no-0.hdf5')

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


