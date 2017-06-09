from __future__ import division, print_function, unicode_literals
from keras.models import model_from_json

import tensorflow as tf
from keras import backend as K
from dataset import Dataset
from constants import (DATA_NAMES, MAP_NAMES, MODELS,
					WEIGHT_NAMES)

DATA_NAME = DATA_NAMES[1]
MAP_NAME = MAP_NAMES[1]
MODEL = MODELS[3]
WEIGHT_NAME = WEIGHT_NAMES[3]


def evaluate_model(model, X_test, Y_test):
	print("Evaluating...")
	everything = model.evaluate(X_test, Y_test)
	loss, accuracy = everything[:2]
	print('\nloss: {} - accuracy: {}'.format(loss, accuracy))


def save_model(model, model_name):
	model_json = model.to_json()
	with open(model_name + ".json", "w") as json_file:
		json_file.write(model_json)

def save_weights(model, weight_name):
	# serialize weights to HDF5
	model.save_weights(weight_name + ".h5")

def load_model(model_name, weight_name=None):
	# load json and create model
	with open(model_name + '.json', 'r') as json_file:
		loaded_model_json = json_file.read()
	loaded_model = model_from_json(loaded_model_json)
	if weight_name:
		# load weights into new model
		loaded_model.load_weights(weight_name + ".h5")
	return loaded_model

def sequence_accuracy(y_true, y_pred):
	return K.mean(K.min(
				tf.to_int32(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))),
				axis=-1)
			)

def compile_model(model):
	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)


if __name__ == "__main__":
	d = Dataset(DATA_NAME, MAP_NAME)
	
	X_test, Y_test = d.get_test_dataset()

	m = load_model(MODEL, WEIGHT_NAME)

	compile_model(m)

	evaluate_model(m, X_test, Y_test)
