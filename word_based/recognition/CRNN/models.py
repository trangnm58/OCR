from __future__ import division, print_function, unicode_literals
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Activation, Dense,
                          Permute, Reshape, Lambda)
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.wrappers import Bidirectional

from utils import Timer
from dataset import (Dataset, MJSynthDataGenerator, get_inputs_outputs)
from constants import *

seed = 13
np.random.seed(seed)

DATA_NAME = DATA_NAMES[0]
MAP_NAME = MAP_NAMES[0]
WEIGHT_NAME = WEIGHT_NAMES[1]


class NN:
    def load_data(self):
        d = Dataset(self.data_name, self.idx_map)
        self.X_train, self.Y_train = d.get_train_dataset()
        self.X_val, self.Y_val = d.get_val_dataset()

        self.X_train, self.Y_train = get_inputs_outputs(self.X_train, self.Y_train)
        self.X_val, self.Y_val = get_inputs_outputs(self.X_val, self.Y_val)

    def train(self, epochs, generator=None):
        initial_epoch = 0
        while True:
            if not generator:
                self.model.fit(self.X_train, self.Y_train,
                               batch_size=self.batch_size,
                               epochs=epochs,
                               initial_epoch=initial_epoch,
                               validation_data=(self.X_val, self.Y_val))
            else:
                self.model.fit_generator(
                    generator=generator.next_train(), validation_data=generator.next_val(),
                    callbacks=[generator], steps_per_epoch=generator.steps_per_epoch,
                    validation_steps=generator.validation_steps,
                    initial_epoch=initial_epoch, epochs=epochs)

            if IS_SERVER:
                break

            initial_epoch = epochs
            epochs = input("More? ")
            if not epochs:
                break
            else:
                epochs = int(epochs) + initial_epoch

    def save_model(self):
        if IS_SERVER:
            self.model.save_weights(self.weight_name + ".h5")
        else:
            name = input("Save weights? [Y/n]: ")
            if not name:
                self.model.save_weights(self.weight_name + ".h5")
            elif name != 'n':
                self.model.save_weights(TRAINED_MODELS + name + ".h5")


class CRNN(NN):
    def __init__(self):
        self.model = None
        self.batch_size = 128
        self.data_name = DATA_NAME
        self.idx_map = MAP_NAME
        self.weight_name = WEIGHT_NAME

        self.height = HEIGHT
        self.width = WIDTH
        self.word_length = MAX_WORD_LENGTH
        self.output_size = OUTPUT_SIZE
        self.input_length = INPUT_LENGTH

        self.test_func = None

    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build(self, dropout=True):
        input_data = Input(name='the_input', shape=(self.height, self.width, 1), dtype='float32')

        inner = Conv2D(filters=32,
                       kernel_size=(5, 5),
                       padding='same',
                       input_shape=(self.height, self.width, 1),
                       activation='relu',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

        inner = Conv2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       activation='relu',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)

        inner = Permute(dims=(2, 1, 3), name='permute')(inner)

        conv_to_rnn_dims = (self.width // 4, (self.height // 4) * 32)

        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        inner = Dense(units=512, activation='relu', name='dense1')(inner)

        inner = Bidirectional(GRU(units=512, return_sequences=True,
                                  activation='relu', dropout=0.5 if dropout else 0.0),
                              merge_mode='sum')(inner)

        inner = Bidirectional(GRU(units=512, return_sequences=True,
                                  activation='relu', dropout=0.5 if dropout else 0.0),
                              merge_mode='sum')(inner)

        inner = Dense(self.output_size, name='dense2')(inner)
        y_pred = Activation('softmax', name='softmax')(inner)

        # Model(inputs=input_data, outputs=y_pred).summary()
        # exit(0)

        labels = Input(name='the_labels', shape=[self.word_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func,
                          output_shape=(1,),
                          name='ctc')([y_pred, labels, input_length, label_length])

        self.model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

        self.test_func = K.function([input_data], [y_pred])

    def run(self, epochs, generator=None, load_data=True, build=True):
        timer = Timer()

        if load_data:
            timer.start("Loading data")
            self.load_data()
            timer.stop()

        if build:
            timer.start("Building model...")
            self.build()
            timer.stop()

        timer.start("Training model...")
        self.train(epochs, generator)
        timer.stop()

        self.save_model()


if __name__ == "__main__":
    m = CRNN()
    generator = MJSynthDataGenerator(m.batch_size)
    m.run(2, generator=generator, load_data=False)
