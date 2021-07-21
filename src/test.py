from __future__ import absolute_import, division, print_function

import os
from gan_tester import GanTester, Splits
from seg_tester import SegTester
from options import TestingOptions

if __name__ == "__main__":
    test_options = TestingOptions()
    opts = test_options.parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.device_num)
    tester = GanTester(opts) if opts.cyclegan else SegTester(opts)
    splits = Splits(opts)

    tester.predict()
    if opts.cyclegan: splits.generate_splits()