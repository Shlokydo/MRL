# Introduction

This implementation is based on the official repository of [CvT](https://github.com/microsoft/CvT).

# Quick start

## Installation

Assuming that you have installed PyTroch and TorchVision, if not, please follow the [officiall instruction](https://pytorch.org/) to install them firstly.
Intall the dependencies using cmd:

``` sh
python -m pip install -r requirements.txt --user -q
```

The code is developed and tested using pytorch 1.9.0. Other versions of pytorch are not fully tested.

## Data preparation

Please prepare the data as following:

``` sh
|-DATASET
  |-imagenet
    |-train
    | |-class1
    | | |-img1.jpg
    | | |-img2.jpg
    | | |-...
    | |-class2
    | | |-img3.jpg
    | | |-...
    | |-class3
    | | |-img4.jpg
    | | |-...
    | |-...
    |-val
      |-class1
      | |-img5.jpg
      | |-...
      |-class2
      | |-img6.jpg
      | |-...
      |-class3
      | |-img7.jpg
      | |-...
      |-...
```

## Run

Each experiment is defined by a yaml config file, which is saved under the directory of `experiments`. The directory of `experiments` has a tree structure like this:

``` sh
experiments
|-{DATASET_A}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_B}
| |-{ARCH_A}
| |-{ARCH_B}
|-{DATASET_C}
| |-{ARCH_A}
| |-{ARCH_B}
|-...
```

We provide a `run.sh` script for running jobs in local machine.

``` sh
Usage: run.sh [run_options]
Options:
  -g|--gpus <1> - number of gpus to be used
  -n|--nodes <1> - number of nodes
  -t|--job-type <aml> - job type (train|test)
```

### Training on local machine

``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/{NAME OF THE CONFIG FILE}
```

You can also modify the config paramters by the command line. For example, if you want to change the lr rate to 0.1, you can run the command:

``` sh
bash run.sh -g 8 -t train --cfg experiments/imagenet/{NAME OF THE CONFIG FILE} TRAIN.LR 0.1
```

Notes:

- The checkpoint, model, and log files will be saved in OUTPUT/{dataset}/{training config} by default.

### Testing pre-trained models

``` sh
bash run.sh -t test --cfg experiments/imagenet/{NAME OF THE CONFIG FILE} TEST.MODEL_FILE ${PRETRAINED_MODLE_FILE}
```
