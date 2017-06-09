from __future__ import division, print_function, unicode_literals
import cv2
from os import listdir
import sys
import json
import numpy as np
import random
import itertools

import model_handler
from dataset import Dataset
from constants import (HEIGHT, WIDTH, MAX_WORD_LENGTH,
					DATA_FOLDERS, MODELS, CHAR_LABEL_MAP,
					LABEL_CHAR_MAP, WEIGHT_NAMES)

DATA_FOLDER = DATA_FOLDERS[0]
MODEL = MODELS[4]
WEIGHT_NAME = WEIGHT_NAMES[4]
TEST_IMAGES = "test_images/"
IMG_EXT = "jpg"


class CRNNRecognizer():
	def __init__(self):
		self.model = None
		self.model_name = MODEL
		self.weight_name = WEIGHT_NAME
		self.data_folder = DATA_FOLDER

		self.height = HEIGHT
		self.width = WIDTH
		self.word_length = MAX_WORD_LENGTH

	def generate_test_data(self, word_list, samples_per_word=1):
		with open(CHAR_LABEL_MAP, 'r', encoding='utf8') as f:
			char_label_map = json.loads(f.read(), encoding='utf8')

		X = np.zeros((0, self.word_length, self.height, self.width), dtype='float32')

		for w in word_list:
			if len(w) > 8:
				continue

			all_imgs = []
			for i in range(len(w)):
				char = w[i]
				l = char_label_map[char]  # label of character ith

				img_names = random.sample(listdir(self.data_folder + l), samples_per_word)
				img_srcs = [self.data_folder + "{}/{}".format(l, n) for n in img_names]
				imgs = [cv2.imread(src, cv2.IMREAD_GRAYSCALE) for src in img_srcs]
				imgs = np.array(imgs) / 255.0
				all_imgs.append(imgs)

			for i in range(len(w), self.word_length):
				imgs = np.zeros((samples_per_word, self.height, self.width), dtype='float32')
				all_imgs.append(imgs)
			
			all_imgs = np.array(all_imgs)
			all_imgs = np.transpose(all_imgs, (1, 0, 2, 3))  # (samples_per_word, self.word_length, self.height, self.width)

			X = np.vstack((X, all_imgs))
		X = X.reshape(X.shape[0], self.word_length, self.height, self.width, 1)
		return X

	# char_imgs: (self.word_length, some_height, some_width)
	def img_to_data(self, char_img_list):
		X = np.zeros((self.word_length, self.height, self.width), dtype='float32')
		for i in range(len(char_img_list)):
			X[i, :, :] = cv2.resize(char_img_list[i], (self.height, self.width), interpolation=cv2.INTER_CUBIC) / 255.0
		return X.reshape(1, self.word_length, self.height, self.width, 1)

	def display_samples(self, X):
		# display images
		display_X = np.vstack((np.hstack((char for char in w)) for w in X))
		cv2.imshow('Sample images', display_X)
		cv2.waitKey(0)

	def predict_word(self, X):
		with open(LABEL_CHAR_MAP, 'r', encoding='utf8') as f:
			label_char_map = json.loads(f.read(), encoding='utf8')

		all_labels = list(label_char_map.keys())
		all_labels.sort()

		self.model = model_handler.load_model(self.model_name, self.weight_name)
		predictions = self.model.predict(X, verbose=0)

		words = []
		for p in predictions:
			w = []
			for c in p:
				idx = np.argmax(c)
				if idx < len(all_labels):
					char = label_char_map[all_labels[idx]]
					w.append(char)
			words.append(''.join(w))
		return words


if __name__ == "__main__":
	m = CRNNRecognizer()
	if len(sys.argv) >= 2:
		names = sys.argv[1:]
		bases = [TEST_IMAGES + n for n in names]
		ext = IMG_EXT

		X = np.zeros((0, m.word_length, m.height, m.width, 1), dtype='float32')
		for b in bases:
			img_names = ["{}_{}.{}".format(b, idx, ext) for idx in range(m.word_length)]
			imgs = [cv2.imread(n, cv2.IMREAD_GRAYSCALE) for n in img_names]
			imgs = [i for i in imgs if i is not None]
			X = np.vstack((X, m.img_to_data(imgs)))
	else:
		X = m.generate_test_data(['trAng', 'chuyEN', 'sAng'])
		
	with open('output.txt', 'w', encoding='utf8') as f:
		f.write('\n'.join(m.predict_word(X)))
	m.display_samples(X)
