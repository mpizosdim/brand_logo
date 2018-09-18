from keras import backend as K
from keras.layers import Input, Dense, Reshape
from random import shuffle
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from PIL import Image
import matplotlib.pyplot as plt


class DCGan(object):
    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.data = None
        self.img_height = None
        self.img_width = None
        self.img_channels = None
        self.generator = None
        self.discriminator = None
        self.model = None

    def preprocess(self, path_of_data, img_width, img_height):
        result = []
        for file in os.listdir(path_of_data):
            logo = img_to_array(load_img(path_of_data + "/" + file, target_size=(img_height, img_width)))
            if np.mean(logo) > 254:
                continue
            logo_image = (logo.astype(np.float32) / 255) * 2 - 1
            result.append(logo_image)
        shape_of_image = list(set([x.shape for x in result]))[0]
        data = np.array(result)
        shuffle(data)
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = shape_of_image[2]
        self.data = data
        print("number of data points: %s" % len(self.data))
        pass

    def _build_generator(self, init_img_width, init_img_height):
        generator_input = Input(shape=(100,))
        generator_layer = Dense(128 * init_img_width * init_img_height)(generator_input)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_height, init_img_width, 128),
                                  input_shape=(128 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)
        return Model(generator_input, generator_output)

    def _build_discriminator(self):
        img_input2 = Input(shape=(self.img_height, self.img_width, self.img_channels,))
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        discriminator_layer = Dense(1)(img_layer2)
        discriminator_output = Activation('sigmoid')(discriminator_layer)
        return Model(img_input2, discriminator_output)

    def build_model(self):
        # ==== build distriminator ====
        self.discriminator = self._build_discriminator()
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])
        print('discriminator: ', self.discriminator.summary())

        # ==== build generator ====
        self.generator = self._build_generator(self.img_width // 4, self.img_height // 4)

        # ==== gan ====
        self.discriminator.trainable = False
        gan_input = Input(shape=(100,))
        gen_output = self.generator(gan_input)
        discr_output = self.discriminator(gen_output)
        self.model = Model(gan_input, discr_output)
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)
        print('GAN: ', self.model.summary())

    def _plot_loss(self, lose_d, lose_g, accuracy_d):
        epochs = np.arange(len(lose_d))
        plt.figure(1)
        plt.subplot(211)
        plt.plot(epochs, lose_d, 'r', epochs, lose_g, 'b')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.subplot(212)
        plt.plot(epochs, accuracy_d)
        plt.show()

    def fit(self, epochs, batch_size, save_image_path=None, output_image_every_n_epochs=100, number_of_images=5):
        d_losses_all = []
        g_losses_all = []
        d_accuracy_all = []
        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            d_accuracies = []
            print("====" * 6)
            print("epoch is: %s" % epoch)
            batch_count = int(self.data.shape[0] / batch_size)
            for batch_index in range(batch_count):
                image_batch = self.data[batch_index * batch_size:(batch_index + 1) * batch_size]
                image_batch = np.array(image_batch)
                noise_batch = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                generated_images = self.generator.predict(noise_batch)
                self.discriminator.trainable = True
                d_loss_real, d_accuracy_real = self.discriminator.train_on_batch(image_batch, np.array([1] * batch_size))
                d_loss_fake, d_accuracy_fake = self.discriminator.train_on_batch(generated_images, np.array([0] * batch_size))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                d_accuracy = 0.5 * np.add(d_accuracy_real, d_accuracy_fake)
                d_losses.append(d_loss)
                d_accuracies.append(d_accuracy)

                self.discriminator.trainable = False
                noise_batch = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                g_loss = self.model.train_on_batch(noise_batch, np.array([1] * batch_size))
                g_losses.append(g_loss)

            d_losses_all.append(np.mean(d_losses))
            g_losses_all.append(np.mean(g_losses))
            d_accuracy_all.append(np.mean(d_accuracies))

            print("d_loss: %f" % np.mean(d_losses))
            print("g_loss: %f" % np.mean(g_losses))
            print("d_accuracy: %f" % np.mean(d_accuracies))

            if epoch % output_image_every_n_epochs == 0:
                for i in range(number_of_images):
                    self.generate_image().show()
                    if save_image_path:
                        self.generate_image().save(save_image_path + "_" + str(i) +"_" + str(epoch) + ".png")

        self._plot_loss(d_losses_all, g_losses_all, d_accuracy_all)

    def generate_image(self):
        noise_batch = np.random.uniform(-1.0, 1.0, size=[1, 100])
        generated_images = self.generator.predict(noise_batch)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))


if __name__ == '__main__':
    epochs = 2
    batch_size = 32
    img_width, img_height = 160, 80
    gan = DCGan()
    gan.preprocess("labeled_data/", img_width, img_height)
    gan.build_model()
    gan.fit(epochs, batch_size, None, 100, 5)

#TODO: split accuracy to real accuracy and fake accuracy.
#TODO: add save function for the model.