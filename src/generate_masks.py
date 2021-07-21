# Set up imports
import os
import cv2
import sys
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted

import torchvision.transforms as transforms

# Imports from endo utils project
sys.path.append("../ext/endo_utils/data_utils/")
from io_utils import *
from process_utils import *


def add_point_to_mask(mask, x_mean, y_mean, func="gaussian", spread=1):
    """
    Adds a gaussian point at (x_mean, y_mean) with variance sigma to mask
    :param mask: numpy array of shape (h, w)
    :param x_mean: float value denoting center_x of gaussian
    :param y_mean: float value denoting center_y of gaussian
    :return: numpy array with gaussian values added around x_mean, y_mean
    """
    y = np.linspace(0, mask.shape[0] - 1, mask.shape[0])
    x = np.linspace(0, mask.shape[1] - 1, mask.shape[1])
    x, y = np.meshgrid(x, y)

    try:
        if func == "tanh":
            drawn_mask = mask + (
                        255 * (1 + (np.tanh(- (np.pi * np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)) / spread))))
        elif func == "gaussian":
            drawn_mask = mask + (
                        255 * np.exp(-((x - x_mean) ** 2 / (2 * spread ** 2) + (y - y_mean) ** 2 / (2 * spread ** 2))))
    except:
        print("Please specify a valid blur function")
        raise AttributeError
    # Euclidean distance function
    drawn_mask[drawn_mask > 255] = 255  # Round off extra values
    return drawn_mask


def labels_to_mask(labels, blur=True, blur_func="gaussian", spread=1):
    mask = np.zeros((labels["imageHeight"], labels["imageWidth"]))
    if labels["points"]:
        # If no points are found, do nothing, return empty mask
        points = [(points["x"], points["y"]) for points in labels["points"]]  # Get points as a list of tuples
        if blur:
            for point in points:
                mask = add_point_to_mask(mask=mask,
                                         x_mean=point[coordinates["x"]],
                                         y_mean=point[coordinates["y"]], func=blur_func, spread=spread)
        else:
            for point in points:
                mask[point[coordinates["y"]], point[coordinates["x"]]] = 1  # point[y,x] in i,j indexing
    # return Image.fromarray(np.uint8(mask)).convert('L')  # alb aug needs PIL image
    return np.uint8(mask)


parser = argparse.ArgumentParser('Generate masks from json files')
parser.add_argument("--dataroot", type=str, help="path where the datasets are located")
parser.add_argument("--binary", help="If set, generates a binary mask without a blur function",
                    action="store_true")
parser.add_argument("--blur_func", type=str,
                    help="If blur, then the blurring function that has to be applied to the point",
                    default="gaussian")
parser.add_argument("--spread", type=int, help="Spread parameter of the blur function", default="2")

if __name__ == '__main__':

    args = parser.parse_args()
    coordinates = {"x": 0, "y": 1}
    op_dirs = get_sub_dirs(args.dataroot)

    for op in op_dirs:  # For a surgery folder path

        videos = get_sub_dirs(op)  # Video folder path
        for video in videos:

            label_list = natsorted(glob.glob(os.path.join(video, "point_labels", "*.json")))  # get all label files
            mask_path = os.path.join(video, "masks")
            succ = check_and_create_folder(mask_path)

            for label_file in tqdm(label_list):
                filename = os.path.splitext(Path(label_file).name)[0]
                labels = json_loader(label_file)  # Get labels from the json file
                mask = labels_to_mask(labels, blur=not(args.binary), blur_func=args.blur_func,
                                      spread=args.spread)  # Create mask from the labels
                cv2.imwrite(os.path.join(mask_path, filename + ".png"), mask)  # Write the mask

    if args.binary:
        print("Successfully saved the binary masks")
    else:
        print("Successfully saved masks with {} blur function, with spread {}".format(str(args.blur_func),
                                                                                      str(args.spread)))