import scipy
import scipy.misc
from PIL import Image, ImageFilter
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pilutil
import random
import imutil

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, use_colab=False, train_edge=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        if not is_testing:
            batch_images = np.random.choice(path, size=batch_size)
        else:
            batch_images = path[:batch_size]

        imgs_A = []
        imgs_B = []
        labels = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = pilutil.imresize(img_A, self.img_res)
            if train_edge: img_A = imutil.make_edge(img_A)
            img_B = pilutil.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            img_path = img_path.split('\\')[-1] if not use_colab else img_path.split('/')[-1]
            if img_path[0] == '0':
                labels.append(0)
            elif img_path[0] == '1':
                labels.append(1)

        if train_edge:  imgs_A = np.array(imgs_A)/255.
        else: imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B, labels

    def load_batch(self, batch_size=1, is_testing=False, add_noise=False, show_dataset=False, use_colab=False, train_edge=False, noise_value=30):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        random.shuffle(path)

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches):
            batch = path[i*batch_size:(i+1)*batch_size]

            imgs_A, imgs_B, labels = [], [], []
            for fn in batch:
                img = pilutil.imread(fn)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = pilutil.imresize(img_A, self.img_res)
                if train_edge:
                    img_A = imutil.make_edge(img_A,10)
                img_B = pilutil.imresize(img_B, self.img_res)

                #noise
                if add_noise:
                    #apply noise
                    noise_range = random.randint(1,noise_value)
                    noise_np = np.random.randint(-noise_range,noise_range,(256,256,3))
                    img_A = img_A + noise_np
                    #apply noise : flip left right
                    if np.random.random() > 0.5 :
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                if show_dataset:
                    o = np.zeros(shape=(256, 512, 3))
                    o[:256, :256,:] = img_A[:256,:256,:]
                    o[:256, 256:,:] = img_B[:256,:256,:]
                    o/=256
                    plt.imshow(o)
                    plt.show()

                #add in batch
                imgs_A.append(img_A)
                imgs_B.append(img_B)
                fn = fn.split('\\')[-1] if not use_colab else fn.split('/')[-1]
                if fn[0] == '0' :
                    labels.append(0)
                else:
                    labels.append(1)

            if train_edge:  imgs_A = np.array(imgs_A)/255.
            else: imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            yield imgs_A, imgs_B, labels


    def imread(self, path):
        return pilutil.imread(path, mode='RGB').astype(np.float)