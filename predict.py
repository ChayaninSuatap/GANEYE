from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pix2pix_eye import Pix2Pix

print('loading model')
o = Pix2Pix(gen_weights_fn='gen.hdf5', dis_weights_fn='dis.hdf5', load_for_predict=True, save_path='saved_model_eyes')
model = o.combined
print('loaded')
#sample image ( compare with validate while training )
# o.sample_images(9999,9999)

#predict 
img = scipy.misc.imread('input.jpg', mode='RGB').astype(np.float)
img = scipy.misc.imresize(img, (256,256))
xs = np.array([img])/127.5 - 1.
print('xs shape', xs.shape)
pred = model.predict( xs)
# print(np.array(pred).shape)
print(pred)
plt.imshow(pred[1][0] * 0.5 + 0.5)
plt.show()


