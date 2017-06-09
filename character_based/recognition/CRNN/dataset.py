from __future__ import division, print_function, unicode_literals
import os
import json
import sys
import cv2
import pickle
import numpy as np
seed = 13
np.random.seed(seed)

from utils import Timer
from constants import (HEIGHT, WIDTH, MAX_WORD_LENGTH,
					SAMPLE_PER_WORD, PICKLE_DATA, DATA_FOLDERS,
					DATA_NAMES, IDX_MAPS, VN_DICT,
					CHAR_LABEL_MAP)

DATA_FOLDER = DATA_FOLDERS[0]
DATA_NAME = DATA_NAMES[0]


class Dataset():
	def __init__(self, data_name, idx_map=None):
		# load dataset from pickle files
		self.X = None
		self.Y = None
		self._load_data(data_name)

		self.train_idx = []
		self.val_idx = []
		self.test_idx = []

		if idx_map:
			# read dataset from files
			self._read_train_val_test_set(idx_map)

			self.train_idx = np.array(self.train_idx, dtype='int')
			self.val_idx = np.array(self.val_idx, dtype='int')
			self.test_idx = np.array(self.test_idx, dtype='int')

	def get_dataset(self):
		return self.X, self.Y

	def get_train_dataset(self):
		return self._get_dataset(self.train_idx)

	def get_val_dataset(self):
		return self._get_dataset(self.val_idx)

	def get_test_dataset(self):
		return self._get_dataset(self.test_idx)

	def create_new_train_val_test_set(self, ratio=(0.8, 0.8)):
		temp = []
		for i in range(self.X.shape[0]):
			r = np.random.random()
			if r < ratio[0]:
				temp.append(i)
			else:
				self.test_idx.append(i)

		for i in temp:
			r = np.random.random()
			if r < ratio[1]:
				self.train_idx.append(i)
			else:
				self.val_idx.append(i)

		np.random.shuffle(self.train_idx)
		np.random.shuffle(self.val_idx)
		np.random.shuffle(self.test_idx)

		# save their indexes to files
		with open(IDX_MAPS + "map_{}".format(self.X.shape[0]), "w") as f:
			f.write(json.dumps({
				"train_idx": self.train_idx,
				"val_idx": self.val_idx,
				"test_idx": self.test_idx
			}))

	def _get_dataset(self, idx_list):
		X = self.X[idx_list]
		Y = self.Y[idx_list]
		return X, Y

	def _load_data(self, data_name):
		with open(data_name, "rb") as f:
			self.X = pickle.load(f)
			self.Y = pickle.load(f)

	def _read_train_val_test_set(self, file_name):
		with open(file_name, "r") as f:
			obj = json.loads(f.read())
		self.train_idx = obj["train_idx"]
		self.val_idx = obj["val_idx"]
		self.test_idx = obj["test_idx"]



def pickle_data(data, data_name):
	print("Saving...")
	with open(data_name, "wb") as f:
		for i in data:
			pickle.dump(i, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_character_list_data(all_labels):
	"""
	pickle images of words
	each example is a list of characters
	"""
	all_labels.sort()

	with open(VN_DICT, 'r', encoding='utf8') as f:
		word_list = f.read().split()

	with open(CHAR_LABEL_MAP, 'r', encoding='utf8') as f:
		char_label_map = json.loads(f.read(), encoding='utf8')

	# Create lowercase, uppercase, camel case for each word
	word_list_all_forms = []
	for w in word_list:
		word_list_all_forms.append(w.lower())
		word_list_all_forms.append(w.upper())
		word_list_all_forms.append(w.title())
		if w not in word_list_all_forms:
			word_list_all_forms.append(w)

	print("Total words: ", len(word_list_all_forms))

	X = np.zeros((0, MAX_WORD_LENGTH, HEIGHT, WIDTH), dtype='float32')
	Y = np.zeros((0, MAX_WORD_LENGTH, len(all_labels) + 1), dtype='int')

	break_points = np.linspace(1, len(word_list_all_forms), 15, endpoint=True, dtype='int')[1:] - 1

	count = -1
	for w in word_list_all_forms:
		count += 1

		if len(w) > MAX_WORD_LENGTH:  # if word is longer than MAX_WORD_LENGTH characters => skip
			continue

		all_imgs = []
		labels = np.zeros((SAMPLE_PER_WORD, MAX_WORD_LENGTH, len(all_labels) + 1))

		for i in range(len(w)):
			char = w[i]
			l = char_label_map[char]  # label of character ith

			img_names = np.random.choice(os.listdir(DATA_FOLDER + l), SAMPLE_PER_WORD, replace=False)
			img_srcs = [DATA_FOLDER + "{}/{}".format(l, n) for n in img_names]
			imgs = [cv2.imread(src, cv2.IMREAD_GRAYSCALE) for src in img_srcs]
			imgs = np.array(imgs) / 255.0

			all_imgs.append(imgs)
			labels[:, i, all_labels.index(l)] = 1

		for i in range(len(w), MAX_WORD_LENGTH):
			all_imgs.append(np.zeros((SAMPLE_PER_WORD, HEIGHT, WIDTH), dtype='float32'))
			labels[:, i, len(all_labels)] = 1

		all_imgs = np.transpose(np.array(all_imgs), (1, 0, 2, 3))  # (SAMPLE_PER_WORD, MAX_WORD_LENGTH, HEIGHT, WIDTH)

		X = np.vstack((X, all_imgs))
		Y = np.vstack((Y, labels))

		if count in break_points:
			print("Saving up to {}...".format(count + 1))
			X = X.reshape(X.shape[0], MAX_WORD_LENGTH, HEIGHT, WIDTH, 1)
			
			pickle_data([X, Y], DATA_NAME + '_temp_' + str(np.where(break_points==count)[0][0]))
			
			X = np.zeros((0, MAX_WORD_LENGTH, HEIGHT, WIDTH), dtype='float32')
			Y = np.zeros((0, MAX_WORD_LENGTH, len(all_labels) + 1), dtype='int')


	print("Merging to single file...")
	X = np.zeros((0, MAX_WORD_LENGTH, HEIGHT, WIDTH, 1), dtype='float32')
	Y = np.zeros((0, MAX_WORD_LENGTH, len(all_labels) + 1), dtype='int')

	for i in range(break_points.shape[0]):
		filename = DATA_NAME + '_temp_' + str(i)
		with open(filename, 'rb') as f:
			X = np.vstack((X, pickle.load(f)))
			Y = np.vstack((Y, pickle.load(f)))
		os.remove(filename)

	pickle_data([X, Y], DATA_NAME)


if __name__ == "__main__":
	t = Timer()
	t.start("Processing...")

	if not os.path.exists(PICKLE_DATA):
		os.makedirs(PICKLE_DATA)

	all_labels = os.listdir(DATA_FOLDER)

	pickle_character_list_data(all_labels)

	# create idx map
	d = Dataset(DATA_NAME)
	d.create_new_train_val_test_set()

	t.stop()