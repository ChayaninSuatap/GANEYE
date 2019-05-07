from keras.models import load_model
import scipy
import matplotlib.pyplot as plt
import numpy as np

print('loading model')
model = load_model('model.hdf5')
print('loaded')
img = scipy.misc.imread('my_tower.jpg', mode='RGB').astype(np.float)
img = scipy.misc.imresize(img, (256,256))
xs = np.array([img])/127.5 - 1.
print('xs shape', xs.shape)
pred = model.predict( xs)
# print(pred.shape)
plt.imshow(pred[1][0] * 0.5 + 0.5)
plt.show()


