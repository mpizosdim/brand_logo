from keras import backend as K
from keras.layers import Input, Dense, LSTM, Reshape, concatenate
from random import shuffle
import numpy as np
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from PIL import Image


class DCGan(object):
    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.biggest_sentence = None
        self.vocab_size = None
        self.data = None
        self.char2int_dict = None
        self.img_height = None
        self.img_width = None
        self.img_channels = None
        self.generator = None
        self.discriminator = None
        self.model = None
        self.names_to_be_tested = None

    @staticmethod
    def _word2ind_padded(word, dictionary, max_pad, vocab_size):
        word2ind = [dictionary[x] for x in word]
        X = sequence.pad_sequences([word2ind], maxlen=max_pad, padding="post")[0]
        return to_categorical(X, vocab_size)

    def preprocess(self, path_of_data, img_width, img_height, verbose=0):
        result = []
        biggest_sentence = 0
        all_chars = ''
        for file in os.listdir(path_of_data):
            brand_text = file.replace('.png', '').lower().strip()
            all_chars = all_chars + brand_text
            if len(brand_text) > biggest_sentence:
                biggest_sentence = len(brand_text)

            logo = img_to_array(load_img(path_of_data + "/" + file, target_size=(img_height, img_width)))
            logo_image = (logo.astype(np.float32) / 255) * 2 - 1
            result.append([logo_image, brand_text])

        chars = sorted(list(set(all_chars)))
        char2int_dic = dict((c, i) for i, c in enumerate(chars))
        vocab_size = len(char2int_dic)
        result = [[x[0], self._word2ind_padded(x[1], char2int_dic, biggest_sentence, vocab_size)] for x in result]
        if verbose != 0:
            print("length of all data: %s" % len(all_chars))
            print("biggest logo name: %s" % biggest_sentence)
            print("vocabulary size: %s" % vocab_size)
            print("length of data: %s" % len(result))

        shape_of_image = list(set([x[0].shape for x in result]))[0]
        data = np.array(result)
        shuffle(data)
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = shape_of_image[2]
        self.char2int_dict = char2int_dic
        self.vocab_size = vocab_size
        self.data = data
        self.biggest_sentence = biggest_sentence
        pass

    def set_testing_data(self, names_to_be_tested):
        self.names_to_be_tested = names_to_be_tested

    def build_model(self, lstm_hidden_size=200):

        init_img_width = self.img_width // 4
        init_img_height = self.img_height // 4

        input_layer_text = Input(shape=(self.biggest_sentence, self.vocab_size,))
        lstm_layer = LSTM(lstm_hidden_size)(input_layer_text)
        text_layer = Dense(1024)(lstm_layer)

        generator_layer = Activation('tanh')(text_layer)
        generator_layer = Dense(128 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_height, init_img_width, 128), input_shape=(128 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)
        self.generator = Model(input_layer_text, generator_output)
        self.generator.compile(loss='mean_squared_error', optimizer="SGD")
        print('generator: ', self.generator.summary())

        input_layer_text2 = Input(shape=(self.biggest_sentence, self.vocab_size,))
        lstm_layer2 = LSTM(lstm_hidden_size)(input_layer_text2)
        text_layer2 = Dense(1024)(lstm_layer2)

        img_input2 = Input(shape=(self.img_height, self.img_width, self.img_channels,))
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)

        merged_layer = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged_layer)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)
        self.discriminator = Model([img_input2, input_layer_text2], discriminator_output)
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
        print('generator-discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, input_layer_text])

        self.model = Model(input_layer_text, model_output)
        self.discriminator.trainable = False

        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        print('generator-discriminator: ', self.model.summary())

    def fit(self, epochs, batch_size):

        for epoch in range(epochs):
            d_losses = []
            g_losses = []
            print("epoch is: %s" % epoch)
            batch_count = int(self.data.shape[0] / batch_size)
            print("number of batches: %s" % batch_count)
            for batch_index in range(batch_count):
                image_label_pair_batch = self.data[batch_index * batch_size:(batch_index + 1) * batch_size]
                image_batch = []
                text_batch = []
                for index in range(batch_size):
                    row = image_label_pair_batch[index]
                    img = row[0]
                    logo_text = row[1]
                    image_batch.append(img)
                    text_batch.append(logo_text)
                image_batch = np.array(image_batch)
                text_batch = np.array(text_batch)

                generated_images = self.generator.predict(text_batch)
                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((image_batch, generated_images)), np.concatenate((text_batch, text_batch))], np.array([1] * batch_size + [0] * batch_size))
                d_losses.append(d_loss)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch(text_batch, np.array([1] * batch_size))
                g_losses.append(g_loss)

            print("Epoch %d d_loss : %f" % (epoch, np.mean(d_losses)))
            print("Epoch %d g_loss : %f" % (epoch, np.mean(g_losses)))
            if self.names_to_be_tested:
                for name in self.names_to_be_tested:
                    self.generate_image_from_text(name).show()

    def generate_image_from_text(self, text):
        encoded_text = np.zeros(shape=(1, self.biggest_sentence, self.vocab_size))
        text = self._word2ind_padded(text, self.char2int_dict, self.biggest_sentence, self.vocab_size)
        encoded_text[0, :, :] = text
        generated_images = self.generator.predict(encoded_text)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))


if __name__ == '__main__':
    epochs = 10
    batch_size = 32
    img_width, img_height = 160, 80
    gan = DCGan()
    gan.preprocess("labeled_data/", img_width, img_height)
    gan.build_model()
    gan.set_testing_data(["prada", "ajk", "carrel", "mun"])
    gan.fit(epochs, batch_size)





