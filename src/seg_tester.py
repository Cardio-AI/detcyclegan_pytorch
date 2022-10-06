from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import torch
import random
import pathlib
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import json
from barbar import Bar

# Imports from current project
import utils
import losses
from models import UNet
from dataloader import MonoDatasetWithMask

# Imports from endo utils project
sys.path.append(os.path.abspath("../ext/endo_utils/data_utils/"))
import process_utils
import io_utils

# process_utils.seed_all(10)  # Seed models, random, and numpy

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


class SegTester:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        """
        Load configs
        """
        self.fold = "fold_" + str(self.opt.fold)
        self.exp_name = pathlib.Path(self.opt.exp_dir).name  # remove fold name
        self.weights_path = os.path.join(self.opt.exp_dir, "model_weights",
                                         "weights_{}".format(str(self.opt.load_epoch)))
        self.config_path = os.path.join(self.opt.exp_dir, "config.json")

        with open(os.path.join(self.config_path), 'r') as configfile:
            self.exp_opts = json.load(configfile)
            print("Loaded experiment configs...")

        """
        Load models
        """
        self.kernel = torch.ones(3, 3).to(self.device)
        self.unet = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        checkpoint = torch.load(os.path.join(self.weights_path, self.exp_opts["model_name"] + ".pt"))
        self.unet.load_state_dict(checkpoint["model_state_dict"])
        self.unet.eval()
        print("Loaded pre-trained Unet for experiment: {}".format(self.exp_opts["model_name"]))

        """
        Load data
        """
        split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, self.fold, "val_files.txt")
        if self.opt.fake:
            split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, "fake_B/train_files.txt")

        self.test_filenames = io_utils.read_lines_from_text_file(split_file_path)
        self.test_dataset = MonoDatasetWithMask(data_root_folder=self.opt.dataroot,
                                                filenames=self.test_filenames,
                                                height=self.exp_opts["height"],
                                                width=self.exp_opts["width"])
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                                          num_workers=self.opt.num_workers, drop_last=False)  # put into dataloader

        """
        Set up pred dirs
        """
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}_{}").format(self.opt.data_split, self.fold)
        if self.opt.fake:
            self.save_path = os.path.join(self.opt.pred_dir, self.exp_name, "{}").format(self.opt.data_split)

        self.pred_folder_names = list(set([file.split()[0]
                                           for file in self.test_filenames]))  # Get all unique 'op/video' names
        self.pred_folder_paths = [os.path.join(self.save_path, folder) for folder in self.pred_folder_names]

        if self.opt.save_pred_points:
            # append op/video/pred_points_02,_03
            self.save_pred_points_paths = [os.path.join(folder, points_path)
                                          for folder in self.pred_folder_paths
                                          for points_path in ['pred_points_02', 'pred_points_03']]
            # In Python3, map function is lazy so you have to consume it with list() else it won't execute
            success_bool = list(map(io_utils.check_and_create_folder, self.save_pred_points_paths))

        if self.opt.save_pred_mask:
            self.save_pred_mask_paths = [os.path.join(folder, mask_path)
                                          for folder in self.pred_folder_paths
                                          for mask_path in ['pred_mask_02', 'pred_mask_03']]
            success_bool = list(map(io_utils.check_and_create_folder, self.save_pred_mask_paths))

        if self.opt.save_annotated:
            self.save_annotated_paths = [os.path.join(folder, annotated_path)
                                          for folder in self.pred_folder_paths
                                          for annotated_path in ['annotated_02', 'annotated_03']]
            success_bool = list(map(io_utils.check_and_create_folder, self.save_annotated_paths))

    def predict(self):
        print("Running prediction on test dataset...")
        metrics = []

        for i, batch in enumerate(Bar(self.test_dataloader), 0):
            image, gt_mask, filename = batch
            image_input, gt_mask = image.to(self.device), gt_mask.to(self.device)

            pred_mask, _ = self.unet(image_input)

            image_np = image.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))
            gt_mask_np = gt_mask.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))
            pred_mask_np = pred_mask.detach().cpu().clone().numpy().transpose((0, 2, 3, 1))

            rel_folder, frame_number, side = self.test_dataset.get_split_filename(filename[0])

            if self.opt.save_pred_points:
                json_name = "{:06d}{}".format(int(frame_number), ".json")
                json_path = os.path.join(self.save_path, rel_folder,
                                              "pred_points_0{}".format(self.side_map[side]),
                                              json_name)
                utils.save_points(pred_mask_np[0, ...], json_path)

            if self.opt.save_pred_mask:
                pred_mask_name = "{:06d}{}".format(int(frame_number), ".png")
                pred_mask_path = os.path.join(self.save_path, rel_folder,
                                              "pred_mask_0{}".format(self.side_map[side]),
                                              pred_mask_name)
                save_image(pred_mask, pred_mask_path)

            if self.opt.save_annotated:
                annotated_image_name = "{:06d}{}".format(int(frame_number), ".png")
                annotated_path = os.path.join(self.save_path, rel_folder,
                                              "annotated_0{}".format(self.side_map[side]),
                                              annotated_image_name)
                save_image(image, annotated_path)
                image_cv = cv2.imread(annotated_path)
                annotated_image = utils.get_annotated(image=image_cv, #Image.fromarray(image_cv),
                                                      gt_mask=gt_mask_np[0, ..., 0],
                                                      pred_mask=pred_mask_np[0, ..., 0])
                cv2.imwrite(annotated_path, annotated_image)

            # Compute metrics and save
            metric = losses.dice_coeff(pred=pred_mask, target=gt_mask)
            metrics.append(metric.item())

        print("Evaluation completed. Metric score: {0:.2f} %".format(np.mean(metrics) * 100))
        print("Saved predictions to: {}".format(os.path.join(self.opt.pred_dir, self.exp_name)))
        print('Successfully wrote annotated images to disk')