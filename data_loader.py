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

    def make_dataset_cache(self, is_testing, use_colab, train_edge,
        train_edge_blur_fn, train_edge_blur_val, normalize=False):
        data_type = 'train' if not is_testing else 'test'
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        random.shuffle(path)

        img_A_cache = []
        img_B_cache = []
        label_cache = []

        for i,fn in enumerate(path):
            print('caching',i)
            #read im and split real and blue
            img = pilutil.imread(fn)
            h, w, _ = img.shape
            half_w = int(w/2)
            img_A = img[:, :half_w, :]
            img_B = img[:, half_w:, :]
            #filter edge
            if train_edge:
                img_A = imutil.make_edge(img_A, blur_val=train_edge_blur_val , blur_fn=train_edge_blur_fn)
            img_A = pilutil.imresize(img_A, self.img_res)
            img_B = pilutil.imresize(img_B, self.img_res)
            #compute label
            fn = fn.split('\\')[-1] if not use_colab else fn.split('/')[-1]
            if fn[0] == '0' :
                label_cache.append(0)
            else:
                label_cache.append(1)
            #append
            img_A_cache.append(img_A)
            img_B_cache.append(img_B)
        #normalize
        if normalize:
            if train_edge: img_A_cache = np.array(img_A_cache)/255.
            else: img_A_cache = np.array(img_A_cache)/127.5 - 1.
            img_B_cache = np.array(img_B_cache)/127.5 - 1.
        return img_A_cache, img_B_cache, label_cache

    def load_batch(self, batch_size, is_testing, add_noise, show_dataset, train_edge, cache, noise_value):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path) / batch_size)
        #get cache
        img_A_cache, img_B_cache, label_cache = cache
        for i in range(self.n_batches):#TODO test load best
            img_A_batch = img_A_cache[i*batch_size:(i+1)*batch_size]
            img_B_batch = img_B_cache[i*batch_size:(i+1)*batch_size]
            label_batch = label_cache[i*batch_size:(i+1)*batch_size]

            imgs_A, imgs_B, labels = [], [], []
            for img_A, img_B, label in zip( img_A_batch, img_B_batch, label_batch):
                #noise
                if add_noise:
                    #apply noise
                    noise_range = random.randint(1,noise_value)
                    noise_np = np.random.randint(-noise_range,noise_range,(self.img_res[0], self.img_res[1],3))
                    img_A = img_A + noise_np
                    #apply noise : flip left right
                    if np.random.random() > 0.5 :
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)
                #show dataset option
                if show_dataset:
                    o = np.zeros(shape=(self.img_res[0], self.img_res[1] * 2, 3))
                    o[:self.img_res[0], :self.img_res[1],:] = img_A[:self.img_res[0],:self.img_res[1],:]
                    o[:self.img_res[0], self.img_res[1]:,:] = img_B[:self.img_res[0],:self.img_res[1],:]
                    o/=256
                    plt.imshow(o)
                    plt.show()
                #add in batch
                imgs_A.append(img_A)
                imgs_B.append(img_B)
                labels.append(label)
            #normalize
            if train_edge: imgs_A = np.array(imgs_A)/255.
            else: imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            #yield batch
            yield imgs_A, imgs_B, np.array(labels)

    def imread(self, path):
        return pilutil.imread(path, mode='RGB').astype(np.float)