from __future__ import print_function, division
import scipy
import imutil
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from keras import backend as K
import matplotlib.pyplot as plt
import random
import imutil

class Pix2Pix():
    def __init__(self, init_epoch=0, gen_weights_fn='', dis_weights_fn='',
    save_path='saved_model', dataset_name='facades', dropout=0,
    load_for_predict=False, img_size=(256,256), validate_num=3):
        from matplotlib.pyplot import rcParams
        rcParams['figure.figsize'] = 14, 8
        #pre setting
        self.init_epoch = init_epoch
        self.save_path = save_path
        self.validate_num = validate_num
        # Input shape
        self.img_rows = img_size[1]
        self.img_cols = img_size[0]
        self.channels = 3
        self.channels_blue = 4
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape_blue = (self.img_rows, self.img_cols, self.channels_blue)

        # Configure data loader
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(dropout=dropout)
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator(dropout=dropout)

        # Input images and their conditioning images
        # img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape_blue)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # Discriminators determines validity of translated images / condition pairs
        self.discriminator.trainable = False
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=img_B, outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        if init_epoch>0 or load_for_predict:
            self.generator.load_weights('%s/%s' % (self.save_path, gen_weights_fn,))
            self.discriminator.load_weights('%s/%s' % (self.save_path, dis_weights_fn,))

    def build_generator(self, dropout=0):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            if dropout_rate:
                d = Dropout(dropout_rate)(d) 
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = BatchNormalization(momentum=0.8)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape_blue)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False, dropout_rate=dropout)
        d2 = conv2d(d1, self.gf*2, dropout_rate=dropout)
        d3 = conv2d(d2, self.gf*4, dropout_rate=dropout)
        d4 = conv2d(d3, self.gf*8, dropout_rate=dropout)
        d5 = conv2d(d4, self.gf*8, dropout_rate=dropout)
        d6 = conv2d(d5, self.gf*8, dropout_rate=dropout)
        d7 = conv2d(d6, self.gf*8, dropout_rate=dropout)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8, dropout_rate=dropout)
        u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=dropout)
        u3 = deconv2d(u2, d4, self.gf*8, dropout_rate=dropout)
        u4 = deconv2d(u3, d3, self.gf*4, dropout_rate=dropout)
        u5 = deconv2d(u4, d2, self.gf*2, dropout_rate=dropout)
        u6 = deconv2d(u5, d1, self.gf  , dropout_rate=dropout)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self, dropout=0):

        def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            if dropout_rate:
                d = Dropout(dropout_rate)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape_blue)

        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False, dropout_rate=dropout)
        d2 = d_layer(d1, self.df*2, dropout_rate=dropout)
        d3 = d_layer(d2, self.df*4, dropout_rate=dropout)
        d4 = d_layer(d3, self.df*8, dropout_rate=dropout)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=None, epoch_interval=None, train_on_colab=False,
        add_noise=False, colab_epoch_interval=None, train_edge=False, noise_value=30, dis_noisy_label=False,
        train_edge_blur_fn=imutil.gaussian_blur, train_edge_blur_val=3):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        # loss array
        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            epoch += self.init_epoch + 1
            for batch_i, (imgs_A, imgs_B, labels) in enumerate(
                self.data_loader.load_batch(
                    batch_size, add_noise=add_noise, use_colab=train_on_colab, train_edge=train_edge,
                    noise_value=noise_value,train_edge_blur_fn=train_edge_blur_fn, train_edge_blur_val=train_edge_blur_val 
                    )
                ):
                #mode blue img (imgs_B)
                imgs_B = self.make_imgb_with_label(imgs_B, labels)
                #  Train Discriminator
                fake_A = self.generator.predict(imgs_B)
                #prepare noisy label
                if dis_noisy_label:
                    noisy_valid = np.full((batch_size,) + self.disc_patch, random.random() * 0.2 + 0.8)
                    noisy_fake = np.full((batch_size,) + self.disc_patch, random.random() * 0.2)
                    #random flip
                    if random.random() >= 0.95: noisy_valid, noisy_fake = noisy_fake, noisy_valid

                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], noisy_valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], noisy_fake)
                else:
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake) 
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #  Train Generator
                g_loss = self.combined.train_on_batch(imgs_B, [valid, imgs_A])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print('',end='\r')
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time), end='')
                #add loss array
                d_losses.append(d_loss[0])
                g_losses.append(g_loss[0])
                plt.clf()
                plt.plot(d_losses, label='discriminator')
                plt.plot(g_losses, label='generator')
                plt.title('epoch ' + str(epoch) + ' batch no ' + str(batch_i))

                # If at save interval => save generated image samples
                if (sample_interval!=None and batch_i % sample_interval == 0) or (epoch_interval!=None and epoch % epoch_interval == 0 and batch_i==0):
                    self.sample_images(epoch, batch_i, train_on_colab=train_on_colab, train_edge=train_edge, train_edge_blur_fn=train_edge_blur_fn,
                        train_edge_blur_val=train_edge_blur_val)
                    plt.savefig(self.save_path + '/loss.png')
                    #save model
                    if not train_on_colab:
                        self.discriminator.save_weights('%s/dis_ep-%d-sample-%d.hdf5' % (self.save_path, epoch, batch_i, ))
                        self.generator.save_weights('%s/gen_ep-%d-sample-%d.hdf5' % (self.save_path, epoch, batch_i, ))
                    else:
                        self.discriminator.save_weights('%s/dis.hdf5' % (self.save_path, ))
                        self.discriminator.save_weights('%s/dis_backup.hdf5' % (self.save_path, ))
                        self.generator.save_weights('%s/gen.hdf5' % (self.save_path, ))                        
                        self.generator.save_weights('%s/gen_backup.hdf5' % (self.save_path, ))
                        #save model with number
                        if colab_epoch_interval != None and epoch % colab_epoch_interval == 0:
                            self.discriminator.save_weights('%s/dis_ep-%d-sample-%d.hdf5' % (self.save_path, epoch, batch_i, ))
                            self.generator.save_weights('%s/gen_ep-%d-sample-%d.hdf5' % (self.save_path, epoch, batch_i, )) 
                    print('\nmodel saved')

    def sample_images(self, epoch, batch_i, train_on_colab, train_edge, train_edge_blur_fn, train_edge_blur_val):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, self.validate_num

        imgs_A, imgs_B, labels = self.data_loader.load_data(batch_size=self.validate_num,
            use_colab=train_on_colab, train_edge=train_edge, train_edge_blur_fn=train_edge_blur_fn,
            train_edge_blur_val=train_edge_blur_val)

        tobepred = self.make_imgb_with_label(imgs_B, labels)
        fake_A = self.generator.predict(tobepred)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        if train_on_colab:
            fn = '%s/validate.png' % (self.save_path,)
        else:
            fn = "images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i)
        fig.savefig(fn)
        plt.close()

    def make_imgb_with_label(self, imgs_B, labels):
        ts = [] ###################### label is arrayyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
        for imgb, label in zip(imgs_B, labels):
            t = np.zeros(shape=self.img_shape_blue, dtype=np.float)
            # print(t.shape, imgb.shape)
            t[:self.img_rows, :self.img_cols, 0:2] = imgb[:self.img_rows, :self.img_cols, 0:2]
            t[:self.img_rows, :self.img_cols, 3] = label
            ts.append(t)
        return np.array(ts)

if __name__ == '__main__':
    gan = Pix2Pix(init_epoch=0,
        dataset_name='eyes512', save_path='saved_model_eyes', dropout=0.2, img_size=(512, 512))
    gan.train(epochs=999, batch_size=1, epoch_interval=1, train_on_colab=False, add_noise=True, train_edge=True,
        noise_value=2, dis_noisy_label=True, train_edge_blur_fn=imutil.median_filter, train_edge_blur_val=31)
