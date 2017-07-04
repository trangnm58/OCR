from __future__ import division, print_function, unicode_literals
import cv2
import numpy as np
import itertools

from models import CRNN
from dataset import Dataset
from constants import *


class CRNNRecognizer:
    def __init__(self):
        self.model = None
        self.weight_name = WEIGHT_NAME
        self.data_name = DATA_NAME
        self.idx_map = MAP_NAME

    @staticmethod
    def display_samples(X):
        # display images
        display_X = np.vstack((img for img in X))
        cv2.imshow('Sample images', display_X)
        cv2.waitKey(0)

    def get_X_test(self, start, end):
        d = Dataset(self.data_name, self.idx_map)
        X_test, _ = d.get_test_dataset()
        return X_test[start:end]

    def predict_word(self, X):
        crnn = CRNN()
        crnn.build(dropout=False)
        crnn.model.load_weights(self.weight_name + ".h5")

        out = crnn.test_func([X])[0]
        ret = []

        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if 0 <= c <= 9:
                    outstr += chr(c + ord('0'))
                elif 10 <= c <= 35:
                    outstr += chr(c - 10 + ord('A'))
                elif 36 <= c <= 61:
                    outstr += chr(c - 36 + ord('a'))
            ret.append(outstr)
        return ret


if __name__ == "__main__":
    DATA_NAME = DATA_NAMES[0]
    MAP_NAME = MAP_NAMES[0]
    WEIGHT_NAME = WEIGHT_NAMES[1]

    m = CRNNRecognizer()
    X = m.get_X_test(150, 160)
    print(m.predict_word(X))
    m.display_samples(X)
