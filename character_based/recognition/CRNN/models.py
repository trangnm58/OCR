from __future__ import division, print_function, unicode_literals
import sys
import numpy as np
seed = 13
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed, Bidirectional

from utils import Timer
from dataset import Dataset
import model_handler
from constants import (HEIGHT, WIDTH, MAX_WORD_LENGTH,
					DATA_NAMES, MAP_NAMES, MODELS,
					WEIGHT_NAMES)

DATA_NAME = DATA_NAMES[1]
MAP_NAME = MAP_NAMES[1]
MODEL = MODELS[4]
WEIGHT_NAME = WEIGHT_NAMES[4]


class NN():
	def load_data(self):
		d = Dataset(self.data_name, self.idx_map)
		self.X_train, self.Y_train = d.get_train_dataset()
		self.X_val, self.Y_val = d.get_val_dataset()

	def train(self, epochs):
		initial_epoch = 0
		while True:
			self.model.fit(self.X_train, self.Y_train,
				  batch_size=self.batch_size,
				  epochs=epochs,
				  initial_epoch=initial_epoch,
				  validation_data=(self.X_val, self.Y_val))
			initial_epoch = epochs
			epochs = input("More? ")
			if not epochs:
				break
			else:
				epochs = int(epochs) + initial_epoch

	def save_model(self):
		name = input("Save model? [Y/n]:  ")
		if not name:
			model_handler.save_model(self.model, self.model_name)
		elif name != 'n':
			model_handler.save_model(self.model, name)

		name = input("Save weights? [Y/n]: ")
		if not name:
			model_handler.save_weights(self.model, self.weight_name)
		elif name != 'n':
			model_handler.save_weights(self.model, name)


class CRNN(NN):
	def __init__(self):
		self.model = None
		self.batch_size = 128
		self.data_name = DATA_NAME
		self.idx_map = MAP_NAME
		self.model_name = MODEL
		self.weight_name = WEIGHT_NAME

		self.num_of_class = None
		self.height = HEIGHT
		self.width = WIDTH
		self.word_length = MAX_WORD_LENGTH

	def build(self):
		self.model = Sequential()
		self.model.add(TimeDistributed(
			Conv2D(filters=16,
				kernel_size=(5, 5),
				padding='same',
				input_shape=(self.height, self.width, 1),
				activation='relu'),
			input_shape=(self.word_length, self.height, self.width, 1)
		))
		self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

		self.model.add(TimeDistributed(
			Conv2D(filters=32,
				kernel_size=(3, 3),
				padding='same',
				input_shape=(self.height, self.width, 1),
				activation='relu'),
			input_shape=(self.word_length, self.height, self.width, 1)
		))
		self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

		self.model.add(TimeDistributed(Flatten()))

		self.model.add(Bidirectional(
			GRU(units=512, return_sequences=True,
				activation='relu', dropout=0.5),
			merge_mode='sum'
		))
		self.model.add(Bidirectional(
			GRU(units=512, return_sequences=True,
			activation='relu', dropout=0.5),
			merge_mode='sum'
		))

		self.model.add(TimeDistributed(
			Dense(self.num_of_class, activation='softmax'),
			input_shape=(self.word_length, 512)
		))
		self.model.summary()
		input()

		model_handler.compile_model(self.model)

	def run(self, build=True):
		timer = Timer()

		timer.start("Loading data")
		self.load_data()
		self.num_of_class = self.Y_train.shape[2]
		timer.stop()
		
		if build:
			timer.start("Building model...")
			self.build()
			timer.stop()

		timer.start("Training model...")
		self.train(10)
		timer.stop()

		self.save_model()


if __name__ == "__main__":
	m = CRNN()

	m.model = model_handler.load_model(MODELS[3])
	model_handler.compile_model(m.model)
	m.run(build=False)

	# m.run()
