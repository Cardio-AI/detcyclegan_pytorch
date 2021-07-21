import os
import argparse
import utils

utils.set_cwd()  # Sets project root as cwd
project_root = os.getcwd()

class TrainingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Training options")

        self.parser.add_argument("--cyclegan",
                                 help="if set, trains the Cycle Gan network",
                                 action="store_true")

        self.parser.add_argument("--unet",
                                 help="if set, trains the Unet network",
                                 action="store_true")

        '''
        Paths
        '''
        self.parser.add_argument("--dataroot",
                                 type=str,
                                 help="path where the datasets are located")

        self.parser.add_argument("--split_dir",
                                 type=str,
                                 help="split directory",
                                 default=os.path.join(project_root, "splits"))

        self.parser.add_argument("--data_split",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="mkr_dataset")

        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(project_root, "experiments"))

        self.parser.add_argument("--dataroot_A",
                                 type=str,
                                 help="path where the dataset from domain A is located")
        self.parser.add_argument("--dataroot_B",
                                 type=str,
                                 help="path where the dataset from domain B is located")

        self.parser.add_argument("--data_split_A",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="sim_dataset")

        self.parser.add_argument("--data_split_B",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="mkr_dataset")

        '''
        Training options
        '''
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="suture_detection")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="mitral",
                                 choices=["mitral"])
        self.parser.add_argument("--np",
                                 action="store_true")
        self.parser.add_argument("--save",
                                 help="if set, saves model for cross validation within exp folder",
                                 action="store_true")
        self.parser.add_argument('--save_freq',
                                 type=int,
                                 default=10,
                                 help='save images and model evey save_freq epochs')
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument('--input_nc',
                                 type=int,
                                 default=3,
                                 help='number of channels of input data')
        self.parser.add_argument('--output_nc',
                                 type=int,
                                 default=3,
                                 help='number of channels of output data')
        self.parser.add_argument('--fold',
                                 type=int,
                                 default=1,
                                 help='CV fold')
        self.parser.add_argument('--aug_prob',
                                 type=float,
                                 default=0.5,
                                 help='Probability to apply image+mask augmentations')

        '''
        Hyperparameters
        '''
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=0.001,
                                 help='initial learning rate')
        self.parser.add_argument('--epoch',
                                 type=int,
                                 default=0,
                                 help='starting epoch')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=30,
                                 help='number of epochs of training')
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=1,
                                 help='size of the batches')
        self.parser.add_argument('--decay_epoch',
                                 type=int,
                                 default=100,
                                 help='epoch to start linearly decaying the learning rate to 0')
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of workers for parallel processing",
                                 default=32)
        self.parser.add_argument("--device_num",
                                 type=int,
                                 help="The number of the GPU to train on",
                                 default=1)
        self.parser.add_argument('--cuda',
                                 action='store_true',
                                 help='use GPU computation')

        # options for unet
        self.parser.add_argument("--load_epoch",
                                 type=int,
                                 help="number of workers for parallel processing",
                                 default=200)

        self.parser.add_argument("--exp_dir",
                                 type=str,
                                 help="Path to experiment that has to be loaded")

        self.parser.add_argument("--pretrained",
                                 action='store_true',
                                 help='Whether to load pretrained model while training unet')

        self.parser.add_argument("--exp_dir_A",
                                 type=str,
                                 help="Path to experiment that has to be loaded")
        self.parser.add_argument("--exp_dir_B",
                                 type=str,
                                 help="Path to experiment that has to be loaded")

        self.parser.add_argument('--fake',
                                 action='store_true',
                                 help='Loads split from fake data without fold')

    def parse(self):
        return self.parser.parse_args()

class TestingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Prediction options")

        self.parser.add_argument("--cyclegan",
                                 help="if set, trains the Cycle Gan network",
                                 action="store_true")

        self.parser.add_argument("--unet",
                                 help="if set, trains the Unet network",
                                 action="store_true")

        '''
        Paths
        '''
        self.parser.add_argument("--dataroot",
                                 type=str,
                                 help="path where the datasets are located")

        self.parser.add_argument("--exp_dir",
                                 type=str,
                                 help="Path to experiment that has to be loaded")

        self.parser.add_argument("--split_dir",
                                 type=str,
                                 help="split directory",
                                 default=os.path.join(project_root, "splits"))
        self.parser.add_argument("--data_split",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="mkr_challenge_lr")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(project_root, "experiments"))

        self.parser.add_argument("--dataroot_A",
                                 type=str,
                                 help="path where the dataset from domain A is located")
        self.parser.add_argument("--dataroot_B",
                                 type=str,
                                 help="path where the dataset from domain B is located")

        self.parser.add_argument("--data_split_A",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="sim_challenge_lr")
        self.parser.add_argument("--data_split_B",
                                 type=str,
                                 help="name of split located in split dir",
                                 default="mkr_challenge_lr")

        self.parser.add_argument("--load_epoch",
                                 type=int,
                                 help="number of workers for parallel processing",
                                 default=200)

        self.parser.add_argument("--pred_dir",
                                 type=str,
                                 help="Directory to save predictions",
                                 default=os.path.join(project_root, "predictions"))
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of workers for parallel processing",
                                 default=8)

        self.parser.add_argument('--fold',
                                 type=int,
                                 default=1,
                                 help='CV fold')

        self.parser.add_argument("--device_num",
                                 type=int,
                                 help="The number of the GPU to train on (Default: Quattro",
                                 default=0)

        self.parser.add_argument("--save_pred_mask",
                                 help="if set, saves fake images of domain A",
                                 action="store_true")
        self.parser.add_argument("--save_pred_points",
                                 help="if set, saves fake images of domain B",
                                 action="store_true")
        self.parser.add_argument("--save_annotated",
                                 help="if set, saves fake cycle recovered images of domain A",
                                 action="store_true")

        self.parser.add_argument("--save_real_A",
                                 help="if set, saves real images of domain A",
                                 action="store_true")
        self.parser.add_argument("--save_real_B",
                                 help="if set, saves real images of domain B",
                                 action="store_true")
        self.parser.add_argument("--save_fake_A",
                                 help="if set, saves fake images of domain A",
                                 action="store_true")
        self.parser.add_argument("--save_fake_B",
                                 help="if set, saves fake images of domain B",
                                 action="store_true")
        self.parser.add_argument("--save_fake_A_cycle",
                                 help="if set, saves fake cycle recovered images of domain A",
                                 action="store_true")
        self.parser.add_argument("--save_fake_B_cycle",
                                 help="if set, saves fake cycle recovered images of domain B",
                                 action="store_true")
        self.parser.add_argument("--save_video",
                                 help="if set, saves concatenated video for both domains",
                                 action="store_true")

        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)

        self.parser.add_argument('--aug_prob',
                                 type=float,
                                 default=0.5,
                                 help='Probability to apply image+mask augmentations')

        self.parser.add_argument('--fake',
                                 action='store_true',
                                 help='Loads split from fake data without fold')

    def parse(self):
        return self.parser.parse_args()
