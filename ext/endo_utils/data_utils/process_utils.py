
# Imports
import os
import sys
import glob
import cv2
import PIL
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from barbar import Bar
from time import sleep
from PIL import Image
from pathlib import Path
from natsort import natsorted


def split_image_top_down(image):
    """Splits an image into top half and bottom half"""
    im_height = image.shape[0]
    assert im_height%2 == 0, "Image height is not even"
    top_image = image[: im_height//2, ...]
    bottom_image = image[im_height//2 :, ...]
    return top_image, bottom_image


def split_image_interlaced(image):
    """Splits a 3-channel image in interlaced format into odd-row and even-row image"""
    even_row_image = image[::2]
    odd_row_image = image[1::2]
    return even_row_image, odd_row_image


def convert_to_numpy_image(image_tensor):
    return image_tensor.numpy().transpose((1, 2, 0))


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sub_dirs(path, sort=True, paths=True):
    """ Returns all the sub-directories in the given path as a list
    If paths flag is set, returns the whole path, else returns just the names
    """
    sub_things = os.listdir(path)  # thing can be a folder or a file
    if sort: sub_things = natsorted(sub_things)
    sub_paths = [os.path.join(path, thing) for thing in sub_things]

    sub_dir_paths = [sub_path for sub_path in sub_paths if os.path.isdir(sub_path)]  # choose only sub-dirs
    sub_dir_names = [os.path.basename(sub_dir_path) for sub_dir_path in sub_dir_paths]

    return sub_dir_paths if paths else sub_dir_names