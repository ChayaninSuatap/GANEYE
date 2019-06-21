from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import itertools
import matplotlib.pyplot as plt
import os
import sys
import pilutil
import numpy as np
import matplotlib.pyplot as plt
import random

class DCGAN():
    def __init__(self, img_rows=28, img_cols=28, img_channels=1, latent_dim=100, dataset_name='eyes512'):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = img_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.dataset_name = dataset_name
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        def add_upsampling_layers(model, filter_n):
            model.add(UpSampling2D())
            model.add(Conv2D(filter_n, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))

        #head of model
        model = Sequential()
        model.add(Dense(256 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 256)))
        add_upsampling_layers(model, 128) # 32
        add_upsampling_layers(model, 128) # 64
        add_upsampling_layers(model, 64) # 128
        # add_upsampling_layers(model, 3) # 256

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        #discriminator head
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        def dataset_generator(batch_size):
            dataset_path = 'datasets/' + self.dataset_name + '/'
            fns = os.listdir(dataset_path)
            for img_path in itertools.cycle(fns):
                img = pilutil.imread(dataset_path + img_path)
                h, w, _ = img.shape
                half_w = int(w/2)
                img = img[:, half_w:, :]
                img = pilutil.imresize(img, self.img_shape)
                #flip
                if np.random.random() > 0.5 :
                    img = np.fliplr(img)
                #normalize
                img = img.astype(np.float32) / 127.5 - 1
                #TODO yield batch
                yield img 
        
        d_losses = []
        g_losses = []
        dataset_gen = dataset_generator(batch_size)
        for epoch in range(epochs):
            #fetch dataset
            imgs = np.array([dataset_gen.__next__()])
            #gen noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            #predict
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            noisy_valid = np.full((batch_size, 1), random.random() * 0.2)
            noisy_fake = np.full((batch_size, 1), random.random() * 0.2 + 0.8)
            #random flip
            if random.random() >= 0.95: noisy_valid, noisy_fake = noisy_fake, noisy_valid
            d_loss_real = self.discriminator.train_on_batch(imgs, noisy_valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, noisy_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            g_loss = self.combined.train_on_batch(noise, noisy_valid)

            # Plot the progress
            #add loss array
            d_losses.append(d_loss[0])
            g_losses.append(g_loss)
            plt.clf()
            plt.plot(g_losses, label='generator')
            plt.plot(d_losses, label='discriminator')
            plt.title('epoch ' + str(epoch))
            print('',end='\r')
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss), end='')

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                plt.draw()
                plt.pause(0.01)
                self.save_imgs(epoch)
                #save model
                self.discriminator.save_weights('t/dis.hdf5')
                self.generator.save_weights('t/gen.hdf5')
                self.discriminator.save_weights('t/dis.backup.hdf5')
                self.generator.save_weights('t/gen.backup.hdf5')

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("t/ep_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    dcgan = DCGAN(img_rows=128, img_cols=128, img_channels=3, dataset_name='eyes512/train')
    dcgan.train(epochs=999999, batch_size=1, save_interval=200)