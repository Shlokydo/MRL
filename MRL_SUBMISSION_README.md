# Information

## Code structure

The code is divide into the following branches, provided as tar.gz files of the same name:

- mrl: Codes for the proposed MRL method. Used for generating results for ImageNet-1K classification experiments.
- commonqkv: Codes for the proposed CommonQKV, detailed in Appendix.
- mrl_commonqkv: Codes utilizing MRL and CommonQKV together. Used to pretrain models for the HoVerNet experiments in Section 4.3.
- mrl_commonqkv_gconv: Codes utilizing MRL + Group Convolutions and CommonQKV together. Used to pretrain models for the HoVerNet experiments in Section 4.3.

The configuration files used for the experiments can be found [experiments/imagenet](experiments/imagenet) for the respective branch. All the configuration files follow the naming procedure for the models as specfied in the paper.

The core change to the [CvT](https://github.com/microsoft/CvT) repository on which this repository builds upon is in the model defination file [cls_cvt.py](lib/models/cls_cvt.py).
