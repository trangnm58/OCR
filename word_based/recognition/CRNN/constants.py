import os


IS_SERVER = "posix" == os.name

HEIGHT = 64
WIDTH = 192
MAX_WORD_LENGTH = 16
OUTPUT_SIZE = 63
INPUT_LENGTH = 46  # used in dataset and models

# Folders' names
DATA = "../../../dataset/"
PICKLE_DATA = "pickle_data/"
IDX_MAPS = "idx_maps/"
TRAINED_MODELS = "trained_models/"

# Model configuration files
DATA_FOLDERS = [DATA + i for i in ["IIIT5K-Word_V3.0/", "mjsynth/90kDICT32px/"]]

DATA_NAMES = [PICKLE_DATA + i for i in ["iiit5k"]]

MAP_NAMES = [IDX_MAPS + i for i in ["map_5000"]]

WEIGHT_NAMES = [TRAINED_MODELS + i for i in ["CNNp2_D_biGRU2_iiit5k", "CNNp2_D_biGRU2_mjsynth"]]
