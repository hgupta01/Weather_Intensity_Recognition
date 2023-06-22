# VARG: An Intensity-labelled Video Weather Recognition Dataset 

## Overview

We provide the implementation used for the evaluation of VARG dataset. We used the TSM implementation and modified it to work with our custom dataset.

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Pretrained Weather Classification Models](#pretrained-models)
- [Training](#training)
- [Testing](#testing)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.13 or higher
- [OpenCV](https://opencv.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pytorchvideo](https://pytorchvideo.org/)

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation
Please follow the following steps for dataset preparation
- Download the VARG dataset using [OSF link](https://osf.io/w6q3t/?view_only=5ff76f00497641a18059657fcd1efcf2) in git folder.
    - select 'dataset' folder and click on 'Download as zip'.
- Open terminal and run following commands to uncompress the dataset.
```bash
    find . -iname '*.zip' -exec sh -c 'unzip -o -d "${0%.*}" "$0"' '{}' ';'
    cd dataset/
    find . -iname '*.zip' -exec sh -c 'unzip -o -d "${0%.*}" "$0"' '{}' ';'
    cd ..
```
- Extract videos into frames for fast reading using following script. This step will take around half an hour
```bash
    python varg_video2frames.py
```
- Convert the labels in required format
```bash
    python varg_labels2TSMformat.py
```

Note that the naive implementation involves large data copying and increases memory consumption during training. It is suggested to use the **in-place** version of TSM to improve speed (see [ops/temporal_shift.py](ops/temporal_shift.py) Line 12 for the details.)

## Pretrained Weather Classification Models
Download trained ResNet18 model for image weather classifications from [drive link](https://drive.google.com/drive/folders/1zMz1RTN28bSiL11ncZa6o09-HbkVVsI4?usp=drive_link) in 'base_trained_model' folder.

## Training 

Run follwing commands to train weather intensity classification using ResNet18 with TSM module:

- To train using ImageNet pretrained models, you can run:

  ```bash
  # You should get TSM_custom_RGB_resnet18_shift8_blockres_avg_segment8_e50_snow_imagenet_frames8_imgsz224.pth
  python main.py custom RGB \
       --arch resnet18 --num_segments 8 \
       --pretrain imagenet --batch-size 32 --epochs 50 --workers 16\
       --shift --shift_div=8 --shift_place=blockres \
       --imgsz 224 --default_transform --eval-freq=1 \
       --weather snow --suffix snow_imagenet_frames8_imgsz224
  ```
- To train using pretrained weights from image weather classification, you can run ::

  ```
  # You should get TSM_custom_RGB_Custom-resnet18_shift8_blockres_avg_segment8_e50_snow_weathernet_frames8_imgsz224.pth
  python main.py custom RGB \
       --arch Custom-resnet18 --num_segments 8 \
       --pretrain base_trained_model/snow_resnet18_224 --batch-size 32 --epochs 50 --workers 16\
       --shift --shift_div=8 --shift_place=blockres \
       --imgsz 224 --default_transform --eval-freq=1 \
       --weather snow --suffix snow_weathernet_frames8_imgsz224
  ```
You can change the weather type (snow, fog or rain), image size (224 or 352), and pretrained weights (imagenet or weathernet) . The suffix is used to maintain the specific name of the experiments.

## Testing 

For example, to test the downloaded pretrained models on test dataset run the following commands. It will show the plot of confusion matrix

```bash
# using Imagenet based trained model
python test.py --num_segments 8 --imgsz 224 --weights imagenet --weather snow --arch resnet18

# using WeatherNet based trained model
python test.py --num_segments 8 --imgsz 224 --weights weathernet --weather snow --arch Custom-resnet18
```

## Citations
We used the TSM code to evaluate VARG dataset
```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
} 
```
