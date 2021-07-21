# detCycleGAN_pytorch

This repo contains the reference implementation of detcyclegan in Pytorch, for the paper

> **Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation**
>
> Lalith Sharan, Gabriele Romano, Sven Koehler, Halvar Kelm, Matthias Karck, Raffaele De Simone, Sandy Engelhardt  
>
> [Accepted, IEEE JBHI 2021](https://arxiv.org/abs/2107.06941)

Please see the [license file](LICENSE) for terms os use of this repo.
If you find our work useful in your research please consider citing our paper:

```
@misc{sharan2021mutually,
      title={Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation},
      author={Lalith Sharan and Gabriele Romano and Sven Koehler and Halvar Kelm and Matthias Karck and Raffaele De Simone and Sandy Engelhardt},
      year={2021},
      eprint={2107.06941},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Setup

A conda environment is recommended for setting up an environment for model training and prediction.
There are two ways this environment can be set up:

1. Cloning conda environment (recommended)
```
conda env create -f detcyclegan.yml
conda activate detcyclegan
```

2. Installing requirements
```
conda intall --file conda_requirements.txt
conda install -c pytorch torchvision=0.7.0
pip install --r requirements.txt
```

### Prediction of suture detection for a single image

You can predict depth for a single image with:
```shell
python test.py --dataroot ~/data/mkr_dataset/ --exp_dir ~/experiments/unet_baseline_fold_1/ --save_pred_points
```
* The command ```save_pred_points``` saves the predicted landmark co-ordinates in the resepective op folders in the ```../predictions``` directory.
* The command ```save_pred_mask``` saves the predicted mask that is the output of the model in the resepective op folders in the ```../predictions``` directory. The final points are extracted from this mask.

### Dataset preparation

You can download the challenge dataset from the synapse platform by signing up for the [AdaptOR 2021 Challenge](https://adaptor2021.github.io/) from the Synapse platform.
* The Challenge data is present in this format: dataroot --> op_date --> video_folders --> images, point_labels
* Generate the masks with a Gaussian likelihood by running the following script:
You can predict depth for a single image with:
```shell
python generate_masks.py --dataroot /path/to/data
```
* Generate the split files for the generated masks, for cross-validation by running the following script:
You can predict depth for a single image with:
```shell
python generate_splits.py --splits_name mkr_dataset --num_folds 4
```

* This repo is inspired by the following repos:
* [CycleGAN PyTorch](https://github.com/aitorzip/PyTorch-CycleGAN)
* [Monodepth2](https://github.com/nianticlabs/monodepth2)
