import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm

calib_file = "data/carla_infos/carla_infos_train_mini.pkl"# 基于终端的相对路径

with open(calib_file, 'rb') as f:
    calib_data = pickle.load(f)
    
#print(calib_data)
