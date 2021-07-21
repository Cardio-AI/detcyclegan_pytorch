# Imports
import os
import sys
import torch
import random
import numpy as np
from torchvision.transforms import functional as func


from torchvision import transforms
from torch.utils.data import Dataset

sys.path.append(os.path.abspath("../../endo_utils/data_utils/"))
import io_utils
import process_utils

seed = 10
np.random.seed(seed)
random.seed(seed)

# Torch seeds
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class MonoDataset(Dataset):
    """ This dataset works with the endoscopic dataset which are in the format:
    Surgery--> Video --> images which contain images

    This mono class, works with split files which are text files that contain the
    relative path and frame number.
    The class reads the image file paths that are specified in the text file and loads the images.
    It applies the specified transforms to the image, else it just converts it into a tensor.
    :returns Pre-processed Image as a tensor
    """

    def __init__(self, data_root_folder=None,
                 filenames=None,
                 height=448,
                 width=448,
                 aug=None,
                 color_aug=None,
                 aug_prob=0.5,
                 image_ext='.png'):
        super(MonoDataset).__init__()
        self.data_root_folder = data_root_folder
        self.filenames = filenames
        self.height = height
        self.width = width
        self.image_ext = image_ext
        self.aug = aug
        self.color_aug = color_aug
        self.aug_prob = aug_prob
        self.resize = transforms.Resize((self.height, self.width))
        self.image_loader = io_utils.pil_loader

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns the image with transforms applied to it"""
        filename, frame_name = self.filenames[index].split()
        image = self.image_loader(os.path.join(self.data_root_folder, filename, "images", frame_name+self.image_ext))
        image = self.preprocess(image)
        return image

    def preprocess(self, image):
        image = self.resize(image)  # Output: Resized PIL Image
        if self.color_aug and random.random() > self.aug_prob: image = self.color_aug(image)
        if self.aug: image = self.aug(image=np.asarray(image))["image"]  # alb needs np input
        return func.to_tensor(image)


class MonoDatasetWithMask(MonoDataset):
    """ Generate masks for the corresponding images and return them
    Masks are present in the folder called 'masks'
    """

    def __init__(self, mask_transform=None,
                 aug=None,
                 image_aug=None,
                 aug_prob=0.5,
                 **kwargs):
        super(MonoDatasetWithMask, self).__init__(**kwargs)
        self.aug = aug
        self.aug_prob = aug_prob
        self.image_aug = image_aug
        self.mask_loader = io_utils.mask_loader
        if mask_transform is None: self.mask_transform = transforms.ToTensor()
        else: self.mask_transform = transforms.Compose(mask_transform)

    def __getitem__(self, index):
        filename, frame_name = self.filenames[index].split()
        image = self.image_loader(os.path.join(self.data_root_folder, filename, "images", frame_name+self.image_ext))
        mask = self.mask_loader(os.path.join(self.data_root_folder, filename, "masks", frame_name+self.image_ext))
        image, mask = self.preprocess_image_mask(image=image, mask=mask)
        return image, mask, self.filenames[index]

    def preprocess_image_mask(self, image, mask):
        image = self.resize(image)  # Output: Resized PIL Image
        if self.image_aug: image = self.image_aug(image=np.asarray(image))["image"]
        if self.aug:
            augmented = self.aug(image=np.asarray(image), mask=np.asarray(mask))
            # alb needs np input
            image = augmented["image"]
            mask = augmented["mask"]
        image = func.to_tensor(np.array(image))
        mask = func.to_tensor(np.array(mask))
        return image, mask


class MonoDatasetWithMaskTwoDomains(MonoDatasetWithMask):
    """ Load unpaired images from two domains
    Together with the mask
    Same image aug is applied to images from both domains
    Similarly, same mask aug applied to masks from both domains
    """

    def __init__(self,
                 data_root_folder_A=None,
                 data_root_folder_B=None,
                 filenames_A=None,
                 filenames_B=None,
                 aligned=False,
                 **kwargs):
        super(MonoDatasetWithMaskTwoDomains, self).__init__(**kwargs)

        self.aligned = aligned
        self.data_root_folder_A = data_root_folder_A
        self.data_root_folder_B = data_root_folder_B
        self.filenames_A = filenames_A
        self.filenames_B = filenames_B
        self.normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __getitem__(self, index):
        # Get the relative path, frame number and side : need for both mask and image
        filename_A, frame_number_A = self.filenames_A[index % len(self.filenames_A)].split()
        # mod only prevents index overshoot that may be caused due to differing lengths returned by max
        if not self.aligned:
            filename_B, frame_number_B = self.filenames_B[random.randint(0, len(self.filenames_B) - 1)].split()
        else: filename_B, frame_number_B = self.filenames_B[index % len(self.filenames_B)].split()

        image_A = self.image_loader(os.path.join(self.data_root_folder_A, filename_A, "images",
                                                 frame_number_A+self.image_ext))
        mask_A = self.mask_loader(os.path.join(self.data_root_folder_A, filename_A, "masks",
                                               frame_number_A+self.image_ext))

        image_B = self.image_loader(os.path.join(self.data_root_folder_B, filename_B, "images",
                                                 frame_number_B+self.image_ext))
        mask_B = self.mask_loader(os.path.join(self.data_root_folder_B, filename_B, "masks",
                                               frame_number_B+self.image_ext))

        image_A, mask_A = self.preprocess_image_mask(image=image_A, mask=mask_A)
        image_B, mask_B = self.preprocess_image_mask(image=image_B, mask=mask_B)
        return image_A, mask_A, image_B, mask_B

    def __len__(self):
        return max(len(self.filenames_A), len(self.filenames_B))