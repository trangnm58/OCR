from __future__ import division, print_function, unicode_literals
import json
import cv2
import pickle
import numpy as np
from scipy.io import loadmat

from utils import Timer
from constants import *

seed = 13
np.random.seed(seed)

DATA_FOLDER = DATA_FOLDERS[0]
DATA_NAME = DATA_NAMES[0]


class Dataset:
    def __init__(self, data_name, idx_map=None):
        # load dataset from pickle files
        self.X = None
        self.Y = None
        self._load_data(data_name)

        if idx_map:
            # read index map from file
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
        train_idx = []
        val_idx = []
        test_idx = []

        temp = []
        for i in range(self.X.shape[0]):
            r = np.random.random()
            if r < ratio[0]:
                temp.append(i)
            else:
                test_idx.append(i)

        for i in temp:
            r = np.random.random()
            if r < ratio[1]:
                train_idx.append(i)
            else:
                val_idx.append(i)

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        # save their indexes to files
        with open(IDX_MAPS + "map_{}".format(self.X.shape[0]), "w") as f:
            f.write(json.dumps({
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx
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


def l_prime_len(length):
    # calculate length of l'
    return length * 2 + 1


def pickle_iiit5k_word():
    """
    pickle images of words in IIIT5K dataset
    each example is a single word (token)
    """

    def text_to_labels(text):
        # <blank> : 0
        # 0 - 9: 1 - 10
        # A - Z: 11 - 36
        # a - z: 37 - 62
        ret = [0]
        for char in text:
            if '0' <= char <= '9':
                ret.append(ord(char) - ord('0') + 1)
            elif 'A' <= char <= 'Z':
                ret.append(ord(char) - ord('A') + 11)
            elif 'a' <= char <= 'z':
                ret.append(ord(char) - ord('a') + 37)
            ret.append(0)

        return np.array(ret)[:l_prime_len(MAX_WORD_LENGTH)]

    traindata = loadmat(DATA_FOLDER + "traindata.mat")['traindata'][0]
    testdata = loadmat(DATA_FOLDER + "testdata.mat")['testdata'][0]
    all_data = np.append(traindata, testdata)

    print("All data: ", all_data.shape)

    m = all_data.shape[0]
    X = np.zeros((m, HEIGHT, WIDTH), dtype='float32')
    Y = np.zeros((m, l_prime_len(MAX_WORD_LENGTH)), dtype='int')

    for i in range(m):
        label = all_data[i][1][0]  # string
        img_src = DATA_FOLDER + all_data[i][0][0]

        img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / 255.0

        X[i] = img
        Y[i, 0:l_prime_len(len(label))] = text_to_labels(label)

    X = X.reshape(X.shape[0], HEIGHT, WIDTH, 1)
    pickle_data([X, Y], DATA_NAME)


if __name__ == "__main__":
    t = Timer()
    t.start("Processing...")

    if not os.path.exists(PICKLE_DATA):
        os.makedirs(PICKLE_DATA)

    pickle_iiit5k_word()

    # create idx map
    d = Dataset(DATA_NAME)
    d.create_new_train_val_test_set()

    t.stop()
