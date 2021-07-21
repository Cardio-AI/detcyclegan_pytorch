
# Imports
import os
import sys
import glob
import cv2
import PIL
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from barbar import Bar
from time import sleep
from PIL import Image
from pathlib import Path
from natsort import natsorted


def write_list_to_text_file(save_path, text_list, verbose=True):
    """
    Function to write a list to a text file.
    Each element of the list is written to a new line.
    Note: Existing text in the file will be overwritten!
    :param save_path: Path to save-should be complete with .txt extension)
    :param text_list: List of text-each elem of list written in new line)
    :param verbose: If true, prints success message to console
    :return: No return, writes file to disk and prints a success message
    """
    with open(save_path, 'w+') as write_file:
        for text in text_list:
            if isinstance(text, str):
                write_file.write(text + '\n')
            else:
                write_file.write(str(text) + '\n')
        write_file.close()
    if verbose: print("Text file successfully written to disk at {}".format(save_path))


def check_and_create_folder(path):
    """Check if a folder in given path exists, if not then create it"""
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else: return False


def print_elements_of_list(list):
    """Prints each element of list in a newline"""
    [print(element) for element in list]


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('RGB')


def mask_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        with Image.open(file) as image:
            return image.convert('L')


def json_loader(path):
    with open(path, 'rb') as file:
        return json.load(file)
    
    
def write_to_json_file(content, path):
    with open(path, 'w') as file:
        json.dump(content, file, indent=4)


def read_lines_from_text_file(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def drawPoints(image, points_list=None, color=(255, 255, 0)):
    """
    Draws a set of points on a given image and returns the drawn image
    :param image: Image on which the points to be drawn
    :param points_list: A list of tuples (x,y)
    :param color: Color of the points to be drawn
    :return: Drawn image
    """
    for point in points_list:
        image = cv2.circle(image, point, radius=0, color=color, thickness=3)
    return image