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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms as transforms

import albumentations as alb
import albumentations.augmentations.transforms as alb_tr
#import nonechucks as nc
import json
from barbar import Bar
import telegram
import emoji

# Imports from current project
import utils
import losses
from models import Generator, Generator_new
from models import Discriminator
from models import UNet
from dataloader import MonoDatasetWithMaskTwoDomains

# Imports from endo utils project
sys.path.append(os.path.abspath("../endo_utils/data_utils/"))
import io_utils
import process_utils

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CombinedTrainer:
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

        """
        Model setup
        """
        #  define gan models
        #self.netG_A2B = Generator(self.opt.input_nc, self.opt.output_nc).to(self.device)
        #self.netG_B2A = Generator(self.opt.output_nc, self.opt.input_nc).to(self.device)
        self.netG_A2B = Generator_new(f=32, blocks=6).to(self.device)
        self.netG_B2A = Generator_new(f=32, blocks=6).to(self.device)
        self.netD_A = Discriminator(self.opt.input_nc).to(self.device)
        self.netD_B = Discriminator(self.opt.output_nc).to(self.device)

        # define unet models
        self.kernel = torch.ones(3, 3).to(self.device)
        self.unet_or = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)
        self.unet_sim = UNet(n_channels=3, n_classes=1, kernel=self.kernel, bilinear=True).to(self.device)

        #  init models
        self.models = {"netG_A2B": self.netG_A2B, "netG_B2A": self.netG_B2A,
                       "netD_A": self.netD_A, "netD_B": self.netD_B}
        [model.apply(utils.weights_init_normal) for model_name, model in self.models.items()]

        #  init unet models trained on sim images
        self.exp_name_A = pathlib.Path(self.opt.exp_dir_A).name[:-7]  # remove fold
        self.weights_path_A = os.path.join(self.opt.exp_dir_A,
                                           "model_weights", "weights_{}".format(str(self.opt.load_epoch)))
        self.unet_sim_checkpoint = torch.load(os.path.join(self.weights_path_A, self.exp_name_A + ".pt"))
        self.unet_sim.load_state_dict(self.unet_sim_checkpoint["model_state_dict"])
        #self.unet_sim.load_state_dict(torch.load(os.path.join(self.weights_path_A, self.exp_name_A + ".pth")))
        self.unet_sim.eval()
        print("Loaded pre-trained Unet from domain A for experiment: {}".format(self.exp_name_A))

        # init unet model trained on OR images
        self.exp_name_B = pathlib.Path(self.opt.exp_dir_B).name[:-7]  # remove fold
        self.weights_path_B = os.path.join(self.opt.exp_dir_B,
                                           "model_weights", "weights_{}".format(str(self.opt.load_epoch)))
        self.unet_or_checkpoint = torch.load(os.path.join(self.weights_path_B, self.exp_name_B + ".pt"))
        self.unet_or.load_state_dict(self.unet_or_checkpoint["model_state_dict"])
        self.unet_or.eval()
        print("Loaded pre-trained Unet from domain B for experiment: {}".format(self.exp_name_B))


        #  init optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                            lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B]

        #  init LR schedulers, only if decay start epoch is within number of epochs to be trained
        if self.opt.num_epochs <= self.opt.decay_epoch: self.lr_decay = False
        else: self.lr_decay = True

        if self.lr_decay:
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                                    lr_lambda=utils.LambdaLR(self.opt.num_epochs,
                                                                                       self.opt.epoch,
                                                                                       self.opt.decay_epoch).step)
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A,
                                                                      lr_lambda=utils.LambdaLR(self.opt.num_epochs,
                                                                                         self.opt.epoch,
                                                                                         self.opt.decay_epoch).step)
            self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B,
                                                                      lr_lambda=utils.LambdaLR(self.opt.num_epochs,
                                                                                         self.opt.epoch,
                                                                                         self.opt.decay_epoch).step)

        """
        Loss functions
        """
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()

        """
        Data setup
        """
        split_file_path_A = os.path.join(self.opt.split_dir, self.opt.data_split_A, self.fold, "{}_files.txt")
        split_file_path_B = os.path.join(self.opt.split_dir, self.opt.data_split_B, self.fold, "{}_files.txt")

        self.train_filenames_A = io_utils.read_lines_from_text_file(split_file_path_A.format("train"))
        self.train_filenames_B = io_utils.read_lines_from_text_file(split_file_path_B.format("train"))

        self.image_aug = alb.Compose([alb.Resize(height=self.opt.height, width=self.opt.width),
                                      #alb_tr.ColorJitter(brightness=0.2,
                                      #                   contrast=(0.3, 1.5),
                                      #                   saturation=(0.5, 2),
                                      #                   hue=0.1,
                                      #                   p=0.5),
                                      alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.image_mask_aug = alb.Compose([alb.Rotate(limit=(-60, 60),
                                                      p=self.opt.aug_prob),
                                           #alb.IAAAffine(translate_percent=10, shear=0.1,
                                           #              p=self.opt.aug_prob),
                                           alb.HorizontalFlip(p=0.5),
                                           alb.VerticalFlip(p=0.5)])

        #self.image_mask_aug = alb.Compose([#alb.RandomCrop(height=self.opt.height, width=self.opt.width),
        #                                       alb.HorizontalFlip(p=0.5)]) #,
        #                                       #alb.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) #,
        #                                       #alb_tr.Resize(height=self.opt.height, width=self.opt.width)])

        # define training dataset
        self.train_dataset = MonoDatasetWithMaskTwoDomains(data_root_folder_A=self.opt.dataroot_A,
                                                           data_root_folder_B=self.opt.dataroot_B,
                                                           filenames_A=self.train_filenames_A,
                                                           filenames_B=self.train_filenames_B,
                                                           height=self.opt.height,
                                                           width=self.opt.width,
                                                           aug=self.image_mask_aug,
                                                           image_aug=self.image_aug,
                                                           aug_prob=self.opt.aug_prob)
        #self.train_dataset = nc.SafeDataset(self.train_dataset)  # remove problematic samples
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size,
                                           shuffle=True, num_workers=self.opt.num_workers,
                                           drop_last=True)  # put into dataloader

        # Save image augmentations to config file
        image_aug_dict = alb.to_dict(self.image_aug) if self.image_aug else None
        image_mask_aug_dict = alb.to_dict(self.image_mask_aug) if self.image_mask_aug else None
        self.aug_dict = {"image_aug": image_aug_dict,
                         "image_mask_aug": image_mask_aug_dict}

        #######################
        self.target_real = torch.ones(self.opt.batch_size, requires_grad=False, device=self.device)
        self.target_fake = torch.zeros(self.opt.batch_size, requires_grad=False, device=self.device)

        self.fake_A_buffer = utils.ReplayBuffer()  # concatenates the fake to feed it back
        self.fake_B_buffer = utils.ReplayBuffer()

    """
    Training and optimisation
    """
    def train(self):
        self.writer = SummaryWriter(self.log_path)  # init tensorboard summary writer
        self.save_configs()  # save script config to json file
        self.append_configs(item=self.aug_dict)

        print("Running experiment named: {} on device:{}...".format(self.folder_name,
                                                                    self.opt.device_num))

        [model.train() for model_name, model in self.models.items()]  # set models to train mode
        for epoch in range(self.opt.num_epochs):
            print("Epoch {}".format(epoch+1))
            time_before_epoch_train = time.time()
            self.run_epoch(epoch+1)  # model computation for 1 epoch

            epoch_train_duration = time.time() - time_before_epoch_train
            if self.lr_decay:
                self.lr_scheduler_G.step()  # update learning rates after 1 epoch
                self.lr_scheduler_D_A.step()
                self.lr_scheduler_D_B.step()

            print('Epoch train time: {} min {} sec'.
                  format(int(epoch_train_duration // 60), int(epoch_train_duration % 60)))
            if (epoch+1) % self.opt.save_freq == 0: self.save_models(epoch+1)  # save model every save_freq epochs

    def run_epoch(self, epoch):
        loss_dict = {}  # Inst. and Init. loss dict
        for key in ["loss_G", "loss_G_identity", "loss_G_GAN", "loss_G_cycle", "loss_D"]:
            loss_dict[key] = 0

        for i, batch in enumerate(Bar(self.train_dataloader), 0):
            # This train loop is adapted from this repo:
            # https://github.com/aitorzip/PyTorch-CycleGAN
            # Get data
            real_A, mask_A, real_B, mask_B = batch

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            mask_A, mask_B = mask_A.to(self.device), mask_B.to(self.device)

            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.set_requires_grad([self.netG_A2B, self.netG_B2A], True)
            self.optimizer_G.zero_grad()
            ###### Generators A2B and B2A ######
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.netG_A2B(real_B)
            loss_identity_B = self.criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = self.netG_B2A(real_A)
            loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0

            # GAN loss for B domain
            fake_B = self.netG_A2B(real_A)
            pred_mask_fake_B, pred_mask_fake_B_gaussian = self.unet_or(0.5 * (fake_B + 1.0))
            pred_fake = self.netD_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, self.target_real)

            # GAN loss for A domain
            fake_A = self.netG_B2A(real_B)
            pred_mask_fake_A, pred_mask_fake_A_gaussian = self.unet_sim(0.5 * (fake_A + 1.0))
            pred_fake = self.netD_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, self.target_real)

            point_loss_fake_B = self.criterion_unet(y_true=mask_A, y_pred=pred_mask_fake_B_gaussian,
                                                    y_pred_binary = pred_mask_fake_B, y_true_binary=mask_A)
            point_loss_fake_A = self.criterion_unet(y_true=mask_B, y_pred=pred_mask_fake_A_gaussian,
                                                    y_pred_binary = pred_mask_fake_A, y_true_binary=mask_B)

            # Cycle loss
            recovered_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = self.netG_A2B(fake_A)
            loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * 10.0

            # Point loss for recovered, with UNet two outputs
            pred_mask_recovered_B, pred_mask_recovered_B_gaussian = self.unet_or(0.5 * (recovered_B + 1.0))
            point_loss_recovered_B = self.criterion_unet(y_true=mask_B, y_pred=pred_mask_recovered_B_gaussian,
                                                         y_pred_binary=pred_mask_recovered_B, y_true_binary=mask_B)

            pred_mask_recovered_A, pred_mask_recovered_A_gaussian = self.unet_sim(0.5 * (recovered_A + 1.0))
            point_loss_recovered_A = self.criterion_unet(y_true=mask_A, y_pred=pred_mask_recovered_A_gaussian,
                                                         y_pred_binary=pred_mask_recovered_A, y_true_binary=mask_A)

            # Cross-domain semantic consistency loss
            pred_mask_or_real_A, pred_mask_or_real_A_gaussian = self.unet_or(0.5 * (real_A + 1.0))
            pred_mask_sim_real_B, pred_mask_sim_real_B_gaussian = self.unet_sim(0.5 * (real_B + 1.0))
            cross_semantic_loss_A = self.criterion_unet(y_true=pred_mask_fake_B_gaussian,
                                                        y_pred=pred_mask_or_real_A_gaussian,
                                                        y_true_binary=pred_mask_fake_B,
                                                        y_pred_binary=pred_mask_or_real_A)
            cross_semantic_loss_B = self.criterion_unet(y_true=pred_mask_fake_A_gaussian,
                                                        y_pred=pred_mask_sim_real_B_gaussian,
                                                        y_true_binary=pred_mask_fake_A,
                                                        y_pred_binary=pred_mask_sim_real_B)

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB \
                     + 1.0 * (point_loss_fake_B + point_loss_fake_A) + 1.0 * (point_loss_recovered_A +
                                                                              point_loss_recovered_B) \
                     + cross_semantic_loss_A + cross_semantic_loss_B


            loss_G.backward()

            self.optimizer_G.step()
            ###################################
            self.set_requires_grad([self.netD_A, self.netD_B], True)

            ###### Discriminator A ######
            self.optimizer_D_A.zero_grad()

            # Real loss
            pred_real = self.netD_A(real_A)
            loss_D_real = self.criterion_GAN(pred_real, self.target_real)

            # Fake loss
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            self.optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            self.optimizer_D_B.zero_grad()

            # Real loss
            pred_real = self.netD_B(real_B)
            loss_D_real = self.criterion_GAN(pred_real, self.target_real)

            # Fake loss
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            self.optimizer_D_B.step()
            ###################################
            # Sum all losses over batches in one epoch
            loss_dict["loss_G"] += loss_G
            loss_dict["loss_D"] += (loss_D_A + loss_D_B)

        for key, value in loss_dict.items():
            loss_dict[key] = value/self.opt.batch_size  # average over batch for one epoch
        self.log_losses(loss_dict, epoch)  # log the losses after every epoch

    def criterion_unet(self, y_true, y_pred, y_pred_binary, y_true_binary):
        normal_loss = self.mse(input=y_pred, target=y_true) + 1 - losses.dice_coeff(pred=y_pred, target=y_true)
        binary_loss = self.mse(input=y_pred_binary, target=y_true_binary) + \
                      1 - losses.dice_coeff(pred=y_pred_binary, target=y_true_binary)
        return 0.5 * normal_loss + 0.5 * binary_loss

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def log_losses(self, loss_dict, epoch):
        """Write an event to the tensorboard events file
        """
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, epoch)

    def log_images(self, image_dict, epoch):
        """Write an event to the tensorboard events file
        """
        for key, value in image_dict.items():
            self.writer.add_image(key, value, epoch)

    def save_models(self, epoch):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "model_weights", "weights_{}".format(epoch))
        os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)
            # ::TODO:: Save optimizer state

    def save_configs(self):
        io_utils.write_to_json_file(content=self.opt.__dict__,
                                    path=os.path.join(self.log_path, "config.json"))
        print("Saving script configs...")

    def append_configs(self, item):
        config_dict = io_utils.json_loader(os.path.join(self.log_path, "config.json"))
        config_dict.update(item)
        io_utils.write_to_json_file(content=config_dict,
                                    path=os.path.join(self.log_path, "config.json"))
