import os


IS_SERVER = "posix" == os.name

HEIGHT = 28
WIDTH = 28
MAX_WORD_LENGTH = 8

SAMPLE_PER_WORD = 3

# Folders' names
DATA = "data/"
PICKLE_DATA = "pickle_data/"
IDX_MAPS = "idx_maps/"
TRAINED_MODELS = "trained_models/"

# Text files
CHAR_LABEL_MAP = DATA + "char_label_map.json"
LABEL_CHAR_MAP = DATA + "label_char_map.json"
VN_DICT = DATA + "vn_token_dict.txt"

# Model configuration files
DATA_FOLDERS = [DATA + i for i in ["chars74k_img_good_full_vie_raw/", "chars74k_img_good_sample_vie/"]]

DATA_NAMES = [PICKLE_DATA + i for i in ["vie_raw_s3w3", "vie_s3w3"]]

MAP_NAMES = [IDX_MAPS + i for i in ["map_16410_622", "map_73938_622"]]

MODEL_NAMES = [TRAINED_MODELS + i for i in [
    "CNNp_GRU2", "CNNp2_GRU2",
    "CNNp2_biGRU2_256_con", "CNNp2_biGRU2_512_con",
    "CNNp2_biGRU2_512_sum", "gru3",
    "gru3small"
]]

WEIGHT_NAMES = [TRAINED_MODELS + i for i in [
    "m0d1", "m1d1", "m2d1",
    "m3d1", "m4d1", "m4d0",
    "gru3d0", "gru3sd0"
]]
