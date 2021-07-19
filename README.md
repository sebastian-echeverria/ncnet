# Neighbourhood Consensus Networks

![](https://www.di.ens.fr/willow/research/ncnet/images/teaser.png)


## About

This is the implementation of the paper "Neighbourhood Consensus Networks" by I. Rocco, M. Cimpoi, R. Arandjelović, A. Torii, T. Pajdla and J. Sivic.

For more information check out the project [[website](http://www.di.ens.fr/willow/research/ncnet/)] and the paper on [[arXiv](https://arxiv.org/abs/1810.10510)].


## Getting started

### Dependencies

The code is implemented using Python 3 and PyTorch 0.3. All dependencies should be included in the standard Anaconda distribution.

### Getting the datasets

The PF-Pascal dataset can be downloaded and unzipped by browsing to the `datasets/pf-pascal/` folder and running `download.sh`.

The IVD dataset (used for training for the InLoc benchmark) can be downloaded by browsing to the `datasets/ivd/` folder and first running `make_dirs.sh` and then `download.sh`.

The InLoc dataset (used for evaluation) an be downloaded by browsing to the `datasets/inloc/` folder and running `download.sh`. 

### Getting the trained models

The trained models trained on PF-Pascal (`ncnet_pfpascal.pth.tar`) and IVD (`ncnet_ivd.pth.tar`) can be dowloaded by browsing to the `trained_models/` folder and running `download.sh`.

### Keypoint transfer demo

The demo Jupyter notebook file `point_transfer_demo.py` illustrates how to evaluate the model and use it for keypoint transfer on the PF-Pascal dataset. For this, previously download the PF-Pascal dataset and trained model as indicated above.

## Training

To train a model, run `train.py` with the desired model architecture and the path to the training dataset.

Eg. For PF-Pascal:

```bash
python main.py --ncons_kernel_sizes 5 5 5 --ncons_channels 16 16 1 --dataset_image_path datasets/pf-pascal --dataset_csv_path datasets/pf-pascal/image_pairs/ 
```

Eg. For InLoc: 

```bash
python main.py --ncons_kernel_sizes 3 3 --ncons_channels 16 1 --dataset_image_path datasets/ivd --dataset_csv_path datasets/ivd/image_pairs/ 
```

## Evaluation

Evaluation for PF-Pascal is implemented in the `eval_pf_pascal.py` file. You can run the evaluation in the following way: 

```bash
python eval_pf_pascal.py --checkpoint trained_models/[checkpoint name]
```

Evaluation for PF-Pascal is implemented in the `eval_inloc.py` file. You can run the evaluation in the following way: 

```bash
python eval_inloc.py --checkpoint trained_models/[checkpoint name]
```

This will generate a series of matches files in the `matches/` folder that then need to be fed to the InLoc evaluation Matlab code. 
In order to run the Matlab evaluation, you first need to clone the [InLoc demo repo](https://github.com/HajimeTaira/InLoc_demo), and download and compile all the required depedencies. Then you can modify the `compute_densePE_NCNet.m` file provided in this repo to indicate the path of the InLoc demo repo, and the name of the experiment (the particular folder name inside `matches/`), and run it to perform the evaluation.


## BibTeX 

If you use this code in your project, please cite our paper:
````
@InProceedings{Rocco18b,
        author       = "Rocco, I. and Cimpoi, M. and Arandjelovi\'c, R. and Torii, A. and Pajdla, T. and Sivic, J."
        title        = "Neighbourhood Consensus Networks",
        booktitle    = "Proceedings of the 32nd Conference on Neural Information Processing Systems",
        year         = "2018",
        }
````

## Fork Changes

This fork of the repo is dockerizing the NCnet project. The main changes include:
- Dependencies have been compiled in the Pipfile file.
- Minor fixes have been required to make the code work with exact dependencies.

The exact dependencies needed are not explicitly stated in any of the repos. Some major dependency notes follow. Note, however, that the containerized version of this system handles most of the versioning issues.
- CUDA is required for ncnet to work. There is a memory leak in PyTorch which eats up all RAM if a CPU is used. Thus, this can only run on an NVIDIA GPU enabled computer, with CUDA installed. (CUDA v9.0 is recommended).
- Python 3.6 or 3.7 is required for the current Python libraries. 3.7 is currently used.

In terms of Python packages, the Pipfile file takes care of the needed versions. If changes are made to the Pipfile file, run `update_lock_reqs.sh` to update the Pipfile.lock and requirements.txt files.

### Container Version Instructions

Pre-requisites:
1. Ensure you are running on a computer with CUDA-compatible GPUs
1. Ensure that the CUDA driver has been installed, as well as the NVIDIA Container toolkit (see https://hub.docker.com/r/nvidia/cuda).
1. Ensure that Docker is installed (19.03 or higher recommended).

To build the container:
1. `bash build_container.sh`

To run the default container:
1. `bash run_container.sh`

TODO: Indicate parameters used to run different things in container (other than the default training code).

### Mughal Paper Changes

This version is integrating the model from the Mughal paper, available here: https://github.com/m-hamza-mughal/Aerial-Template-Matching . Major changes in this fork:
- The `lib/model.py` file from the Mughal repo has been used to overwrite the default one.
- The `soft_argmax.py` file from this repo (https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py#L117) has been added, as requested in the Mughal repo.
- Some code fixes have been required on the two files above to make the code work.

Some general notes Python dependencies notes below:
- Torch version 1.0.0 (at least) is needed to support some methods used in model.py that are not in previous versions. (Since the Mughal code was last modified on May/19, it can't have used Torch higher than 1.1).
- Torchvision version has to be the appropriate one to match the Torch version according to this table: https://pypi.org/project/torchvision/

#### Mughal Dataset
The dataset available at https://github.com/m-hamza-mughal/aerial-template-matching-dataset contains annotations for the different images in a csv file. However, the format is confusing since 1) it is not self-documented, nor documented in detail in the paper, and 2) there is no code available that loads this dataset. However, the following can be understood from the paper, images, repo and csv file:

- The dataset contains 2052 images from 3 different areas, and 3 orthomosaics, one per area.
    - Images 1 to 1200 are from NUST area.
    - Images 1201 to 1680 are from DHA area.
    - Images 1681 to 2052 are from the GujarKhan area.
- The labels.csv file has 2052 lines; thus, each line seems to correspond to one image.
- The information in the labels.csv file is called "annotations" in the paper.
- Each image has at most 16 associated point-to-point correspondences between that image and the corresponding orthomosaic.
- Each correspondence has 2 points, and each point has 2 coordinates; thus, each correspondence has 4 values (which we will assume are ximage, yimage, xmosaic, ymosaic).
- Thus, each line in the dataset corresponds to a series of correspondences for that image and its mosaic, in groups of 4 numbers each (in fact, the number of columns for every row seems to be a multiple of 4).
    - However, even though the paper says that each image has at most 16 correspondences, there are lines in the file with up to 192 (768 colunms) correspondences.
- The range of values for each coordinate go between 1 and 893. It is not clear what each point means.. neither pixels nor lat/long coordinates make sense.
- It is also not clear why some images have integer points, while others have decimal ones. It may come from how they were hand labelled.
- Finally, it is not clear how the images are GPS-tagged. The JPEG images do not have metadata for lat/long.
- The TIFF orthomosaic images have embedded GPS information using GeoTIFF in the WGS 84 coordinate system. The gdalinfo tool (installable like this: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html#install-gdal-ogr) can be used to get the files GPS information. Using the Python gdal library, GPS information could be calculated for each pixel if needed.
