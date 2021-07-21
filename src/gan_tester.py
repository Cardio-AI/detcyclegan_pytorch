from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import time
import glob
import torch
import random
import pathlib
import torchsummary
import numpy as np
import itertools
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import albumentations.augmentations.transforms as alb_tr

import albumentations as alb
import nonechucks as nc
import json
from barbar import Bar
import telegram
import emoji
from natsort import natsorted

# Imports from current project
import utils
import losses
from models import Generator, Generator_new_cp, Generator_new
from models import Discriminator
from dataloader import MonoDatasetWithMaskTwoDomains, MonoDatasetWithMask

# Imports from endo utils project
sys.path.append(os.path.abspath("../endo_utils/data_utils/"))
import io_utils
import process_utils

#process_utils.seed_all(10)  # Seed models, random, and numpy

seed = 10
#print("[ Using Seed : ", seed, " ]")

np.random.seed(seed)
random.seed(seed)

# Torch seeds
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GanTester:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        # Set up video params
        self.fps = 2
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        """
        Load configs
        """
        self.fold = "fold_" + str(self.opt.fold)
        self.exp_name = pathlib.Path(self.opt.exp_dir).name
        self.model_path = os.path.join(self.opt.exp_dir, "model_weights", "weights_{}".format(str(self.opt.load_epoch)))
        self.config_path = os.path.join(self.opt.exp_dir, "config.json")

        with open(os.path.join(self.config_path), 'r') as configfile:
            exp_opts = json.load(configfile)
            print("Loaded experiment configs...")

        """
        Load models
        """
        #self.netG_A2B = Generator(exp_opts['input_nc'], exp_opts['output_nc']).to(self.device)  # init models
        #self.netG_B2A = Generator(exp_opts['output_nc'], exp_opts['input_nc']).to(self.device)

        self.netG_A2B = Generator_new(f=32, blocks=6).to(self.device)
        self.netG_B2A = Generator_new(f=32, blocks=6).to(self.device)

        self.netG_A2B.load_state_dict(torch.load(os.path.join(self.model_path, "netG_A2B.pth")))  # load state dicts
        self.netG_B2A.load_state_dict(torch.load(os.path.join(self.model_path, "netG_B2A.pth")))

        self.netG_A2B.eval()  #set model to eval mode
        self.netG_B2A.eval()

        """
        Load data
        """
        split_file_path_A = os.path.join(self.opt.split_dir, self.opt.data_split_A, self.fold, "{}_files.txt")
        split_file_path_B = os.path.join(self.opt.split_dir, self.opt.data_split_B, self.fold, "{}_files.txt")

        self.test_filenames_A = io_utils.read_lines_from_text_file(split_file_path_A.format("val"))
        self.test_filenames_B = io_utils.read_lines_from_text_file(split_file_path_B.format("val"))

        # define training dataset for domain A
        self.test_dataset_A = MonoDatasetWithMask(data_root_folder=self.opt.dataroot_A,
                                          filenames=self.test_filenames_A,
                                          height=self.opt.height,
                                          width=self.opt.width)
        #self.test_dataset_A = nc.SafeDataset(self.test_dataset_A)  # remove problematic samples
        self.test_dataloader_A = DataLoader(self.test_dataset_A, batch_size=1, shuffle=False,
                                            num_workers=self.opt.num_workers, drop_last=False)

        self.test_dataset_B = MonoDatasetWithMask(data_root_folder=self.opt.dataroot_B,
                                          filenames=self.test_filenames_B,
                                          height=self.opt.height,
                                          width=self.opt.width)
        #self.test_dataset_B = nc.SafeDataset(self.test_dataset_B)  # remove problematic samples
        self.test_dataloader_B = DataLoader(self.test_dataset_B, batch_size=1, shuffle=False,
                                            num_workers=self.opt.num_workers, drop_last=False)

        """
        Setup pred paths
        """
        self.save_path_dict = {}
        self.pred_split_fold = "{}_{}_{}".format(self.opt.data_split_A, self.opt.data_split_B, self.fold)

        for key in ["real_A", "real_B", "fake_A", "fake_B", "fake_A_cycle", "fake_B_cycle"]:
            self.save_path_dict[key] = os.path.join(self.opt.pred_dir, self.exp_name, self.pred_split_fold, key)
            #io_utils.check_and_create_folder(self.save_path_dict[key])

        if self.opt.save_real_A:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["real_A"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["real_A"], "mask_02"))
        if self.opt.save_real_B:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["real_B"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["real_B"], "mask_02"))

        if self.opt.save_fake_A:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_A"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_A"], "mask_02"))
        if self.opt.save_fake_B:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_B"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_B"], "mask_02"))

        if self.opt.save_fake_A_cycle:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_A_cycle"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_A_cycle"], "mask_02"))
        if self.opt.save_fake_B_cycle:
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_B_cycle"], "image_02"))
            io_utils.check_and_create_folder(os.path.join(self.save_path_dict["fake_B_cycle"], "mask_02"))

    def predict(self):
        print("Running prediction on test dataset of Domain A...")

        # Save domain A
        for i, batch in enumerate(Bar(self.test_dataloader_A), 0):
            real_A, mask_A, _ = batch
            real_A, mask_A = real_A.to(self.device), mask_A.to(self.device)

            fake_B = self.netG_A2B(real_A)
            fake_A_cycle = self.netG_B2A(fake_B)

            # Input images are not stdised so no need to de-stdise
            # Just de-stdise the predictions
            fake_B, fake_A_cycle = map((lambda x: 0.5*(x + 1.0)), [fake_B, fake_A_cycle])

            # save source images and mask also!!
            # save image files
            if self.opt.save_real_A:
                save_image(real_A, os.path.join(self.save_path_dict["real_A"], "image_02", "{:06d}.png".format(i + 1)))
                save_image(mask_A, os.path.join(self.save_path_dict["real_A"], "mask_02", "{:06d}.png".format(i + 1)))

            if self.opt.save_fake_B:
                save_image(fake_B, os.path.join(self.save_path_dict["fake_B"], "image_02", "{:06d}.png".format(i + 1)))
                save_image(mask_A, os.path.join(self.save_path_dict["fake_B"], "mask_02", "{:06d}.png".format(i + 1)))

            if self.opt.save_fake_A_cycle:
                save_image(fake_A_cycle, os.path.join(self.save_path_dict["fake_A_cycle"], "image_02",
                                                      "{:06d}.png".format(i + 1)))
                save_image(mask_A, os.path.join(self.save_path_dict["fake_A_cycle"], "mask_02",
                                                "{:06d}.png".format(i + 1)))

        # Save domain B
        print("Running prediction on test dataset of Domain B...")
        for i, batch in enumerate(Bar(self.test_dataloader_B), 0):
            # Get data
            real_B, mask_B, _ = batch
            real_B, mask_B = real_B.to(self.device), mask_B.to(self.device)

            fake_A = self.netG_B2A(real_B)
            fake_B_cycle = self.netG_A2B(fake_A)

            # Input images are not stdised so no need to de-stdise
            # Just de-stdise the predictions
            fake_A, fake_B_cycle = map((lambda x: 0.5*(x + 1.0)), [fake_A, fake_B_cycle])

            # save source images and mask also!!
            # save image files
            if self.opt.save_real_B:
                save_image(real_B, os.path.join(self.save_path_dict["real_B"], "image_02", "{:06d}.png".format(i + 1)))
                save_image(mask_B, os.path.join(self.save_path_dict["real_B"], "mask_02", "{:06d}.png".format(i + 1)))

            if self.opt.save_fake_A:
                save_image(fake_A, os.path.join(self.save_path_dict["fake_A"], "image_02", "{:06d}.png".format(i + 1)))
                save_image(mask_B, os.path.join(self.save_path_dict["fake_A"], "mask_02", "{:06d}.png".format(i + 1)))

            if self.opt.save_fake_B_cycle:
                save_image(fake_B_cycle, os.path.join(self.save_path_dict["fake_B_cycle"], "image_02",
                                                      "{:06d}.png".format(i + 1)))
                save_image(mask_B, os.path.join(self.save_path_dict["fake_B_cycle"], "mask_02",
                                                "{:06d}.png".format(i + 1)))

        if self.opt.save_video:
            video_path = os.path.join(self.opt.pred_dir, self.exp_name, self.pred_split_fold, self.exp_name+"_domain_{}.mp4")

            domain_A_videowriter = cv2.VideoWriter(video_path.format("A"), self.fourcc, self.fps,
                                                   (3 * self.opt.width, self.opt.height))
            domain_B_videowriter = cv2.VideoWriter(video_path.format("B"), self.fourcc, self.fps,
                                                   (3 * self.opt.width, self.opt.height))

            self.write_to_video(videowriter_object=domain_A_videowriter,
                                domain_list=["real_A", "fake_B", "fake_A_cycle"])

            self.write_to_video(videowriter_object=domain_B_videowriter,
                                domain_list=["real_B", "fake_A", "fake_B_cycle"])


        print("Saved predictions to: {}".format(os.path.join(self.opt.pred_dir, self.exp_name)))

    def write_to_video(self, videowriter_object, domain_list):
        image_filepaths = natsorted(
            glob.glob(os.path.join(self.opt.pred_dir, self.exp_name, self.pred_split_fold,
                                   domain_list[0], 'image_02', '*.png')))

        frame_names = [os.path.basename(path) for path in image_filepaths]
        image_path = os.path.join(self.opt.pred_dir, self.exp_name, self.pred_split_fold,
                                  "{}", "image_02", "{}")

        for image_name in frame_names:  # 000000.png
            left_image = cv2.imread(image_path.format(domain_list[0], str(image_name)))
            middle_image= cv2.imread(image_path.format(domain_list[1], str(image_name)))
            right_image = cv2.imread(image_path.format(domain_list[2], str(image_name)))

            concat_a = cv2.hconcat([left_image, middle_image, right_image])
            videowriter_object.write(concat_a)

        print(' Successfully wrote video with domains {}, {}, and {} to disk'.format(domain_list[0], domain_list[1],
                                                                                     domain_list[2]))


class Splits:
    def __init__(self, options=None):
        self.opt = options
        self.fold = "fold_" + str(self.opt.fold)
        self.exp_name = pathlib.Path(self.opt.exp_dir).name
        self.pred_split_fold = "{}_{}_{}".format(self.opt.data_split_A, self.opt.data_split_B, self.fold)

    def get_filenames_in_folder_as_list(self, domain):
        """
        Get the filenames in split format: rel path <space> frame number
        Here the rel path is the experiment name
        Under the experiment name lie the fake_A, fake_B folders
        This is for one particular folder out of fake_A, fake_B, etc.
        """
        image_filepaths = natsorted(glob.glob(os.path.join(self.opt.pred_dir, self.exp_name, self.pred_split_fold,
                                                           domain, 'image_02', '*.png')))

        # Prepare text file information
        rel_path_name = os.path.join(self.exp_name, self.pred_split_fold, domain)  # Folder name
        frame_indices = [os.path.basename(os.path.splitext(path)[0]) for path in image_filepaths]

        indices = [' '.join((rel_path_name, frame_index)) for frame_index in frame_indices]
        return indices

    def write_splits(self, domain):
        filenames = self.get_filenames_in_folder_as_list(domain)
        #random.shuffle(filenames)

        f_writepath = os.path.join(self.opt.split_dir, self.exp_name, self.pred_split_fold, domain)
        io_utils.check_and_create_folder(f_writepath)
        io_utils.write_list_to_text_file(save_path=os.path.join(f_writepath, "train_files.txt"),
                                         text_list=filenames, verbose=False)

        print("Extracted {} prediction files from domain {} and wrote them to disk at {}".
              format(len(filenames), domain, f_writepath))

    def generate_splits(self):
        if self.opt.save_fake_A: self.write_splits("fake_A")
        if self.opt.save_fake_B: self.write_splits("fake_B")

        if self.opt.save_fake_A_cycle: self.write_splits("fake_A_cycle")
        if self.opt.save_fake_B_cycle: self.write_splits("fake_B_cycle")

