# detCycleGAN_pytorch

This repo contains the reference implementation of detcyclegan in Pytorch, for the paper

> **Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation**
>
> Lalith Sharan, Gabriele Romano, Sven Koehler, Halvar Kelm, Matthias Karck, Raffaele De Simone, Sandy Engelhardt  
>
> [Accepted, IEEE JBHI 2021](https://ieeexplore.ieee.org/document/9496194)

Please see the [license file](LICENSE) for terms os use of this repo.
If you find our work useful in your research please consider citing our paper:

```
@article{sharan_mutually_2021,
	title = {Mutually improved endoscopic image synthesis and landmark detection in unpaired image-to-image translation},
	issn = {2168-2208},
	doi = {10.1109/JBHI.2021.3099858},
	journal = {IEEE Journal of Biomedical and Health Informatics},
	author = {Sharan, Lalith and Romano, Gabriele and Koehler, Sven and Kelm, Halvar and Karck, Matthias and De Simone,
	Raffaele and Engelhardt, Sandy},
	year = {2021},
	note = {Conference Name: IEEE Journal of Biomedical and Health Informatics},
	keywords = {CycleGAN, Generative adversarial networks, Generative Adversarial Networks, Landmark Detection, Landmark
	Localization, Maintenance engineering, Mitral Valve Repair, Semantics, Surgery, Surgical Simulation, Surgical
	Training, Task analysis, Training, Valves},
	pages = {1--1},
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
