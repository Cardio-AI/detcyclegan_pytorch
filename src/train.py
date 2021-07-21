from __future__ import absolute_import, division, print_function

import os
import time
from utils import get_hrs_min_sec
from options import TrainingOptions
from seg_trainer import SegmentationTrainer
from gan_trainer import TranslationTrainer
from trainer import CombinedTrainer

if __name__ == "__main__":
    train_options = TrainingOptions()
    opts = train_options.parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.device_num)

    # Logic for which model to train
    if opts.cyclegan and opts.unet:
        print("Choose either of cyclegan or unet, or neither of them")
        raise ValueError
    if opts.cyclegan: trainer = TranslationTrainer(opts)
    elif opts.unet: trainer = SegmentationTrainer(opts)
    else: trainer = CombinedTrainer(opts)

    before_train_time = time.time()
    trainer.train()
    train_duration = time.time() - before_train_time

    hrs, min, sec = get_hrs_min_sec(train_duration)
    print('Total train duration: {} hrs {} min {} sec'.format(hrs, min, sec))
