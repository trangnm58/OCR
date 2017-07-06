from __future__ import division, print_function, unicode_literals
import json
import cv2
import pickle
import numpy as np
import keras.callbacks
from scipy.io import loadmat

from utils import Timer
from constants import *

seed = 13
np.random.seed(seed)


def pickle_data(data, data_name):
    print("Saving...")
    with open(data_name, "wb") as f:
        for i in data:
            pickle.dump(i, f, protocol=pickle.HIGHEST_PROTOCOL)


def text_to_labels(text):
    # <blank> : 62
    # 0 - 9: 0 - 9
    # A - Z: 10 - 35
    # a - z: 36 - 61
    ret = []
    for char in text:
        if '0' <= char <= '9':
            ret.append(ord(char) - ord('0'))
        elif 'A' <= char <= 'Z':
            ret.append(ord(char) - ord('A') + 10)
        elif 'a' <= char <= 'z':
            ret.append(ord(char) - ord('a') + 36)
    return np.array(ret)[:MAX_WORD_LENGTH]


def get_inputs_outputs(X, Y):
    size = X.shape[0]
    the_input = X
    the_labels = Y
    input_length = np.ones((size, 1), dtype='int64') * INPUT_LENGTH
    label_length = (Y < 62).sum(axis=1)
    inputs = {
        'the_input': the_input,
        'the_labels': the_labels,
        'input_length': input_length,
        'label_length': label_length
    }
    outputs = {'ctc': np.zeros([size])}
    return inputs, outputs


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


class MJSynthDataGenerator(keras.callbacks.Callback):
    def __init__(self, batch_size):
        self.data_folder = DATA_FOLDERS[1]
        self.batch_size = batch_size

        self.cur_train_idx = 0
        self.cur_val_idx = 0

        with open(self.data_folder + "annotation_train.txt") as f:
            train_list = [line.split(" ")[0] for line in f.read().strip().split("\n")]
        with open(self.data_folder + "annotation_val.txt") as f:
            val_list = [line.split(" ")[0] for line in f.read().strip().split("\n")]

        self.train_data = np.array(train_list)
        self.val_data = np.array(val_list)

        # self.steps_per_epoch = self.train_data.shape[0] // self.batch_size
        # self.validation_steps = self.val_data.shape[0] // self.batch_size

        self.steps_per_epoch = 10000
        self.validation_steps = 6000

    def get_batch(self, index, size, src_type):
        X = np.zeros((0, HEIGHT, WIDTH), dtype='float32')
        Y = np.ones((0, MAX_WORD_LENGTH), dtype='int')

        if src_type == 0:
            data_src = self.train_data
        elif src_type == 1:
            data_src = self.val_data

        for i in range(size):
            label = data_src[index + i].split("_")[1]  # string
            img_src = self.data_folder + data_src[index + i]

            img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
            if img is None:  # some images are broken
                continue
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / 255.0
            X = np.vstack((X, [img]))
            temp = np.ones(MAX_WORD_LENGTH, dtype='int') * 62
            temp[0:len(label)] = text_to_labels(label)
            Y = np.vstack((Y, [temp]))

        X = X.reshape(X.shape[0], HEIGHT, WIDTH, 1)
        return get_inputs_outputs(X, Y)

    def next_train(self):
        while True:
            ret = self.get_batch(self.cur_train_idx, self.batch_size, src_type=0)
            self.cur_train_idx += self.batch_size

            yield ret

    def next_val(self):
        while True:
            ret = self.get_batch(self.cur_val_idx, self.batch_size, src_type=1)
            self.cur_val_idx += self.batch_size

            yield ret

    def on_epoch_begin(self, epoch, logs=None):
        self.cur_train_idx = 0
        self.cur_val_idx = 0
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.val_data)


def pickle_iiit5k_word(data_folder, dataname):
    """
    pickle images of words in IIIT5K dataset
    each example is a single word (token)
    """
    traindata = loadmat(data_folder + "traindata.mat")['traindata'][0]
    testdata = loadmat(data_folder + "testdata.mat")['testdata'][0]
    all_data = np.append(traindata, testdata)

    print("All data: ", all_data.shape)

    m = all_data.shape[0]
    X = np.zeros((m, HEIGHT, WIDTH), dtype='float32')
    Y = np.ones((m, MAX_WORD_LENGTH), dtype='int') * 62

    for i in range(m):
        label = all_data[i][1][0]  # string
        img_src = data_folder + all_data[i][0][0]

        img = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / 255.0

        X[i] = img
        Y[i, 0:len(label)] = text_to_labels(label)

    X = X.reshape(X.shape[0], HEIGHT, WIDTH, 1)
    pickle_data([X, Y], dataname)


if __name__ == "__main__":
    t = Timer()
    t.start("Processing...")

    if not os.path.exists(PICKLE_DATA):
        os.makedirs(PICKLE_DATA)

    # pickle_iiit5k_word(DATA_FOLDERS[0], DATA_NAMES[0])

    # create idx map
    # d = Dataset(DATA_NAMES[0])
    # d.create_new_train_val_test_set()

    t.stop()
