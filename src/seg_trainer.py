from __future__ import absolute_import, division, print_function

import os
import sys
import time
import torch
import random
import torchsummary
import numpy as np
import itertools
import pathlib
from PIL import Image
import torchviz

# Imports from endo utils project
sys.path.append(os.path.abspath("../endo_utils/data_utils/"))
import io_utils
import process_utils

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from kornia.morphology import erosion, dilation
from kornia.geometry import conv_soft_argmax2d
from kornia.losses import tversky_loss, js_div_loss_2d, kl_div_loss_2d
import torchvision.transforms as transforms

import albumentations as alb
import albumentations.augmentations.transforms as alb_tr
#import nonechucks as nc
import json
from barbar import Bar
import telegram
import emoji
from natsort import natsorted

# Imports from current project
import utils
import losses
from models import UNet, Generator
from dataloader import MonoDatasetWithMask



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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SegmentationTrainer:
    def __init__(self, options=None):
        self.opt = options
        self.device = torch.device('cuda:0')

        """
        Callbacks
        """
        self.fold = "fold_" + str(self.opt.fold)
        # If save flag is set, then no timestamp is added, because then same exp name can be used for all folds
        suffix = "_{}".format(str(utils.getTimeStamp())) if not self.opt.save else ""
        self.folder_name = "{}_{}{}".format(self.opt.model_name, self.fold, suffix)

        self.log_path = os.path.join(self.opt.log_dir, self.folder_name)
        if os.path.exists(self.log_path):
            print(self.log_path)
            raise FileExistsError
        self.writer = SummaryWriter(self.log_path)  # init tensorboard summary writer

        """
        Model setup
        """
        self.kernel = torch.ones(3, 3).to(self.device)
        self.model = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        #  init optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                                       patience=10,
                                                                       min_lr=1e-10,
                                                                       factor=0.1)

        if self.opt.pretrained:  # Load pre-trained unet model
            self.exp_name = pathlib.Path(self.opt.exp_dir).name[:-7]  # remove fold
            self.checkpoint_path = os.path.join(self.opt.exp_dir,
                                               "model_weights", "weights_{}".format(str(self.opt.load_epoch)))
            self.checkpoint = torch.load(os.path.join(self.checkpoint_path, self.exp_name + ".pt"))

            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            self.model.train()
            print("Loaded pre-trained Unet from experiment: {}".format(self.exp_name))

        else:
            self.model = utils.init_net(self.model, type="kaiming", mode="fan_in",
                                        activation_mode="relu",
                                        distribution="normal")

        """
        Loss functions
        """
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.dice_loss = losses.dice_coeff_loss
        self.metric_fn = losses.dice_coeff

        """
        Data setup
        """
        self.image_ext = '.npy' if self.opt.np else '.png'
        split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, self.fold, "{}_files.txt")
        if self.opt.fake:
            split_file_path = os.path.join(self.opt.split_dir, self.opt.data_split, "{}_files.txt")

        #self.train_filenames = natsorted(io_utils.read_lines_from_text_file(split_file_path.format("train")))
        self.train_filenames = io_utils.read_lines_from_text_file(split_file_path.format("train"))

        self.image_aug = alb.Compose([alb.Resize(height=self.opt.height, width=self.opt.width),
                                      alb_tr.ColorJitter(brightness=0.2,
                                                         contrast=(0.3, 1.5),
                                                         saturation=(0.5, 2),
                                                         hue=0.1,
                                                         p=0.5)])

        self.image_mask_aug = alb.Compose([alb.Rotate(limit=(-60, 60),
                                                      p=self.opt.aug_prob),
                                           alb.IAAAffine(translate_percent=10, shear=0.1,
                                                         p=self.opt.aug_prob),
                                           alb.HorizontalFlip(p=0.5),
                                           alb.VerticalFlip(p=0.5)]#,
                                          #additional_targets={"binary_mask" : "mask"}
                                           #alb.RandomCrop(height=144, width=256)]
        )

        #self.color_aug = transforms.ColorJitter(brightness=0.2,
        #                                        contrast=(0.3, 1.5),
        #                                        saturation=(0.5, 2),
        #                                        hue=0.1)

        self.in_memory = True if self.opt.in_memory else False


        # define training dataset
        self.train_dataset = MonoDatasetWithMask(data_root_folder=self.opt.dataroot,
                                                 filenames=self.train_filenames,
                                                 height=self.opt.height,
                                                 width=self.opt.width,
                                                 aug=self.image_mask_aug,
                                                 image_aug=self.image_aug,
                                                 aug_prob=self.opt.aug_prob,
                                                 image_ext=self.image_ext)
        #self.train_dataset = nc.SafeDataset(self.train_dataset)  # remove problematic samples
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=self.opt.num_workers,
                                           drop_last=True, worker_init_fn=seed_worker)  # put into dataloader

        self.val = False
        #self.val = True if os.path.exists(split_file_path.format("val")) else False

        if self.val:
            self.val_filenames = io_utils.read_lines_from_text_file(split_file_path.format("val"))
            self.val_dataset = MonoDatasetWithMask(data_root_folder=self.opt.dataroot,
                                                   filenames=self.val_filenames,
                                                   height=self.opt.height,
                                                   width=self.opt.width,
                                                   image_ext=self.image_ext)
            #self.val_dataset = nc.SafeDataset(self.val_dataset)  # remove problematic samples
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size,
                                             shuffle=True, num_workers=self.opt.num_workers,
                                             drop_last=True, worker_init_fn=seed_worker)  # put into dataloader
        #######################

        # Save image augmentations to config file
        self.aug_dict = {"image_aug": alb.to_dict(self.image_aug),
                         "image_mask_aug": alb.to_dict(self.image_mask_aug)}

    """
    Training and optimisation
    """
    def train(self):

        self.save_configs()  # save script config to json file
        self.append_configs(item=self.aug_dict)

        print("Running experiment named: {} on device:{}...".format(self.folder_name,
                                                                    self.opt.device_num))

        for epoch in range(self.opt.num_epochs):
            print("Epoch {}".format(epoch + 1))

            self.model.train()
            time_before_epoch_train = time.time()
            train_loss, train_metric, train_pred = self.compute_epoch(dataloader=self.train_dataloader, train=True)
            self.lr_scheduler.step(train_loss)

            torch.cuda.synchronize()
            epoch_train_duration = time.time() - time_before_epoch_train
            self.log_losses('train_loss', train_loss, epoch + 1)
            self.log_losses('train_metric', train_metric, epoch + 1)
            self.log_images('train', train_pred, epoch + 1)
            print('Epoch {} mean batch train loss: {:0.5f} | train metric: {:0.4f} | epoch train time: {:0.2f}s'.
                  format(epoch + 1, train_loss, train_metric, epoch_train_duration))

            if self.val:
                self.model.eval()
                with torch.no_grad():
                    time_before_epoch_val = time.time()
                    val_loss, val_metric, val_pred = self.compute_epoch(dataloader=self.val_dataloader, train=False)

                epoch_val_duration = time.time() - time_before_epoch_val
                self.log_losses('val_loss', val_loss, epoch + 1)
                self.log_losses('val_metric', val_metric, epoch + 1)
                self.log_images('val', val_pred, epoch + 1)
                print('Epoch {} mean batch val loss: {:0.5f} | val metric: {:0.4f} | epoch val time: {:0.2f}s'.
                      format(epoch + 1, val_loss, val_metric, epoch_val_duration))

            #if (epoch + 1) % self.opt.save_freq == 0: self.save_model(epoch + 1)  # save model every save_freq epoch
            # save model checkpoint every save_freq epochs
            if (epoch + 1) % self.opt.save_freq == 0: self.save_checkpoint(epoch=epoch + 1,
                                                                           loss=train_loss)


    def compute_epoch(self, dataloader, train=True):
        running_loss = 0
        running_metric = 0

        for i, batch in enumerate(Bar(dataloader), 0):
            # Get data
            image, mask, filename = batch
            image, mask = image.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()  # set the gradients to zero
            binary, pred_mask = self.model(image)
            loss = self.compute_losses(y_pred=pred_mask, y_true=mask, y_pred_binary = binary, y_true_binary=mask)

            if train:
                loss.backward()  # backward pass
                self.optimizer.step()  # Update parameters

            metric = self.metric_fn(pred=pred_mask, target=mask)
            running_metric += metric.detach() * self.opt.batch_size
            running_loss += loss.detach() * self.opt.batch_size  # Mean of one batch times the batch size

        epoch_loss = running_loss.item() / len(dataloader.dataset)  # Sum of all samples over number of samples in dataset
        epoch_metric = (running_metric.item() * 100) / len(dataloader.dataset)
        return epoch_loss, epoch_metric, pred_mask[0]

    def log_losses(self, name, loss, epoch):
        """Write an event to the tensorboard events file"""
        self.writer.add_scalar(name, loss, epoch)

    def log_images(self, name, loss, epoch):
        """Write an image to the tensorboard events file"""
        self.writer.add_image(name, loss, epoch)

    def save_model(self, epoch):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "model_weights", "weights_{}".format(epoch))
        os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "{}.pth".format(self.opt.model_name))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

    def save_checkpoint(self, epoch, loss):
        """ Save model weights and optim state to disk
        """
        save_folder = os.path.join(self.log_path, "model_weights", "weights_{}".format(epoch))
        os.makedirs(save_folder)
        save_path = os.path.join(save_folder, "{}.pt".format(self.opt.model_name))
        checkpoint = {'epoch': epoch,
                   'model_state_dict': self.model.state_dict(),
                   'optimizer_state_dict': self.optimizer.state_dict(),
                   'loss': loss}
        torch.save(checkpoint, save_path)

    def save_configs(self):
        io_utils.write_to_json_file(content=self.opt.__dict__,
                                    path=os.path.join(self.log_path, "config.json"))
        print("Saving script configs...")

    def append_configs(self, item):
        config_dict = io_utils.json_loader(os.path.join(self.log_path, "config.json"))
        config_dict.update(item)
        io_utils.write_to_json_file(content=config_dict,
                                    path=os.path.join(self.log_path, "config.json"))

    def compute_losses(self, y_true, y_pred , y_pred_binary, y_true_binary):

        normal_loss = self.mse(input=y_pred, target=y_true) + 1 - losses.dice_coeff(pred=y_pred, target=y_true)
        binary_loss = self.mse(input=y_pred_binary, target=y_true_binary) + \
                      1 - losses.dice_coeff(pred=y_pred_binary, target=y_true_binary)
        return 0.5 * normal_loss + 0.5 * binary_loss

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
