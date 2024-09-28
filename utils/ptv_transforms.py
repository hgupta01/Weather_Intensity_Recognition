from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from pytorchvideo.transforms.transforms import OpSampler
from pytorchvideo.transforms.augmentations import AugmentTransform

from pytorchvideo.transforms import (
    ConvertUint8ToFloat,
    Normalize,
    Permute,
    RandomResizedCrop,
    RandomShortSideScale,
    ShortSideScale,
)
from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225

# A dictionary that contains transform names (key) and their corresponding maximum
# transform magnitude (value).
_TRANSFORM_RANDAUG_MAX_PARAMS = {
    "AdjustBrightness": (1, 0.9),
    "AdjustContrast": (1, 0.9),
    "AdjustSaturation": (1, 0.9),
    "AdjustSharpness": (1, 0.9),
    "AutoContrast": None,
    "Equalize": None,
    "Posterize": (2, 2),
    "TranslateX": (0, 0.25),
    "TranslateY": (0, 0.25),
}

# Hyperparameters for sampling magnitude.
# sampling_data_type determines whether uniform sampling samples among ints or floats.
# sampling_min determines the minimum possible value obtained from uniform
# sampling among floats.
# sampling_std determines the standard deviation for gaussian sampling.
SAMPLING_RANDAUG_DEFAULT_HPARAS = {
    "sampling_data_type": "int",
    "sampling_min": 0,
    "sampling_std": 0.5,
}

_RANDOM_RESIZED_CROP_DEFAULT_PARAS = {
    "scale": (0.6, 1.0),
    "aspect_ratio": (3.0 / 4.0, 4.0 / 3.0),
}


class RandAugment:
    """
    This implements RandAugment for video. Assume the input video tensor with shape
    (T, C, H, W).

    RandAugment: Practical automated data augmentation with a reduced search space
    (https://arxiv.org/abs/1909.13719)
    """

    def __init__(
        self,
        magnitude: int = 9,
        num_layers: int = 2,
        prob: float = 0.5,
        transform_hparas: Optional[Dict[str, Any]] = None,
        sampling_type: str = "gaussian",
        sampling_hparas: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        This implements RandAugment for video.

        Args:
            magnitude (int): Magnitude used for transform function.
            num_layers (int): How many transform functions to apply for each
                augmentation.
            prob (float): The probablity of applying each transform function.
            transform_hparas (Optional[Dict[Any]]): Transform hyper parameters.
                Needs to have key fill. By default, it uses transform_default_hparas.
            sampling_type (str): Sampling method for magnitude of transform. It should
                be either gaussian or uniform.
            sampling_hparas (Optional[Dict[Any]]): Hyper parameters for sampling. If
                gaussian sampling is used, it needs to have key sampling_std. By
                default, it uses SAMPLING_RANDAUG_DEFAULT_HPARAS.
        """
        assert sampling_type in ["gaussian", "uniform"]
        sampling_hparas = sampling_hparas or SAMPLING_RANDAUG_DEFAULT_HPARAS
        if sampling_type == "gaussian":
            assert "sampling_std" in sampling_hparas

        randaug_fn = [
            AugmentTransform(
                transform_name,
                magnitude,
                prob=prob,
                transform_max_paras=_TRANSFORM_RANDAUG_MAX_PARAMS,
                transform_hparas=transform_hparas,
                sampling_type=sampling_type,
                sampling_hparas=sampling_hparas,
            )
            for transform_name in list(_TRANSFORM_RANDAUG_MAX_PARAMS.keys())
        ]
        self.randaug_fn = OpSampler(randaug_fn, num_sample_op=num_layers)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Perform RandAugment to the input video tensor.

        Args:
            video (torch.Tensor): Input video tensor with shape (T, C, H, W).
        """
        return self.randaug_fn(video)
    

def _get_augmentation(aug_type: str, aug_paras: Optional[Dict[str, Any]] = None) -> List[Callable]:
    """
    Initializes a list of callable transforms for video augmentation.

    Args:
        aug_type (str): Currently supports 'default', 'randaug', or 'augmix'.
            Returns an empty list when aug_type is 'default'. Returns a list
            of transforms containing RandAugment when aug_type is 'randaug'
            and a list containing AugMix when aug_type is 'augmix'.
        aug_paras (Dict[str, Any], optional): A dictionary that contains the necessary
            parameters for the augmentation set in aug_type. If any parameters are
            missing or if None, default parameters will be used. Default is None.

    Returns:
        aug (List[Callable]): List of callable transforms with the specified augmentation.
    """

    if aug_paras is None:
        aug_paras = {}

    if aug_type == "default":
        aug = []
    elif aug_type == "randaug":
        aug = [
                Permute((1, 0, 2, 3)),
                RandAugment(magnitude=aug_paras["magnitude"],
                            num_layers=aug_paras["num_layers"],
                            prob=aug_paras["prob"]),
                Permute((1, 0, 2, 3)),
            ]
    else:
        raise NotImplementedError

    return aug


def create_video_transform(
    mode: str,
    convert_to_float: bool = True,
    video_mean: Tuple[float, float, float] = IMAGENET_MEAN,
    video_std: Tuple[float, float, float] = IMAGENET_STD,
    min_size: int = 256,
    max_size: int = 320,
    crop_size: Union[int, Tuple[int, int]] = 224,
    horizontal_flip_prob: float = 0.5,
    aug_type: str = "default",
    aug_paras: Optional[Dict[str, Any]] = None,
    random_resized_crop: bool = False,
    format = "CTHW"
) -> Union[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
]:
    """
    Function that returns a factory default callable video transform, with default
    parameters that can be modified. The transform that is returned depends on the
    ``mode`` parameter: when in "train" mode, we use randomized transformations,
    and when in "val" mode, we use the corresponding deterministic transformations.
    Depending on whether ``video_key`` is set, the input to the transform can either
    be a video tensor or a dict containing ``video_key`` that maps to a video
    tensor. The video tensor should be of shape (C, T, H, W).

                       "train" mode                                 "val" mode
                            
                   (RandAugment/AugMix)                         (ConvertUint8ToFloat)
                            ↓                                           ↓
                  (ConvertUint8ToFloat)                             Normalize
                            ↓                                           ↓
                        Normalize                           ShortSideScale+CenterCrop               
                            ↓                                           
    RandomResizedCrop/RandomShortSideScale+RandomCrop
                            ↓
                   RandomHorizontalFlip

    (transform) = transform can be included or excluded in the returned
                  composition of transformations

    Args:
        mode (str): 'train' or 'val'. We use randomized transformations in
            'train' mode, and we use the corresponding deterministic transformation
            in 'val' mode.
        convert_to_float (bool): If True, converts images from uint8 to float.
            Otherwise, leaves the image as is. Default is True.
        video_mean (Tuple[float, float, float]): Sequence of means for each channel to
            normalize to zero mean and unit variance. Default is (0.45, 0.45, 0.45).
        video_std (Tuple[float, float, float]): Sequence of standard deviations for each
            channel to normalize to zero mean and unit variance.
            Default is (0.225, 0.225, 0.225).
        min_size (int): Minimum size that the shorter side is scaled to for
            RandomShortSideScale. If in "val" mode, this is the exact size
            the the shorter side is scaled to for ShortSideScale.
            Default is 256.
        max_size (int): Maximum size that the shorter side is scaled to for
            RandomShortSideScale. Default is 340.
        crop_size (int or Tuple[int, int]): Desired output size of the crop for RandomCrop
            in "train" mode and CenterCrop in "val" mode. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made. Default is 224.
        horizontal_flip_prob (float): Probability of the video being flipped in
            RandomHorizontalFlip. Default value is 0.5.
        aug_type (str): Currently supports 'default', 'randaug', or 'augmix'. No
            augmentations other than RandomShortSideScale and RandomCrop area performed
            when aug_type is 'default'. RandAugment is used when aug_type is 'randaug'
            and AugMix is used when aug_type is 'augmix'. Default is 'default'.
        aug_paras (Dict[str, Any], optional): A dictionary that contains the necessary
            parameters for the augmentation set in aug_type. If any parameters are
            missing or if None, default parameters will be used. Default is None.
        random_resized_crop_paras (Dict[str, Any], optional): A dictionary that contains
            the necessary parameters for Inception-style cropping. This crops the given
            videos to random size and aspect ratio. A crop of random size relative to the
            original size and a random aspect ratio is made. This crop is finally resized
            to given size. This is popularly used to train the Inception networks. If any
            parameters are missing or if None, default parameters in
            _RANDOM_RESIZED_CROP_DEFAULT_PARAS will be used. If None, RandomShortSideScale
            and RandomCrop will be used as a fallback. Default is None.

    Returns:
        A factory-default callable composition of transforms.
    """

    if isinstance(crop_size, int):
        assert crop_size <= min_size, "crop_size must be less than or equal to min_size"
    elif isinstance(crop_size, tuple):
        assert (
            max(crop_size) <= min_size
        ), "the height and width in crop_size must be less than or equal to min_size"
    else:
        raise TypeError

    random_resized_crop_paras = None
    if random_resized_crop:
        random_resized_crop_paras = {}
        random_resized_crop_paras["target_height"] = crop_size
        random_resized_crop_paras["target_width"] = crop_size
        if "scale" not in random_resized_crop_paras:
            random_resized_crop_paras["scale"] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS["scale"]
        if "aspect_ratio" not in random_resized_crop_paras:
            random_resized_crop_paras["aspect_ratio"] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS["aspect_ratio"]

    transform = Compose(
        (
            _get_augmentation(aug_type=aug_type, aug_paras=aug_paras) if mode == "train" else []
        )
        + ([ConvertUint8ToFloat()] if convert_to_float else [])
        + [Normalize(mean=video_mean, std=video_std)]
        + (
            (
                [RandomResizedCrop(**random_resized_crop_paras)]
                if random_resized_crop_paras is not None
                else [
                    RandomShortSideScale(
                        min_size=min_size,
                        max_size=max_size,
                    ),
                    RandomCrop(size=crop_size),
                ]
                + [RandomHorizontalFlip(p=horizontal_flip_prob)]
            )
            if mode == "train"
            else [
                ShortSideScale(size=min_size),
                CenterCrop(size=crop_size),
            ]
        )
        +([] if (format == "CTHW") else [Permute((1, 0, 2, 3))])
    )

    return transform


def create_video_optflow_transform(
    mode: str,
    convert_to_float: bool = True,
    video_mean: Tuple[float, float, float] = IMAGENET_MEAN,
    video_std: Tuple[float, float, float] = IMAGENET_STD,
    min_size: int = 256,
    max_size: int = 320,
    crop_size: Union[int, Tuple[int, int]] = 224,
    horizontal_flip_prob: float = 0.5,
    aug_type: str = "default",
    aug_paras: Optional[Dict[str, Any]] = None,
    random_resized_crop: bool = False,
    format = "CTHW"
) -> Union[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
]:

    if isinstance(crop_size, int):
        assert crop_size <= min_size, "crop_size must be less than or equal to min_size"
    elif isinstance(crop_size, tuple):
        assert (
            max(crop_size) <= min_size
        ), "the height and width in crop_size must be less than or equal to min_size"
    else:
        raise TypeError

    random_resized_crop_paras = None
    if random_resized_crop:
        random_resized_crop_paras = {}
        random_resized_crop_paras["target_height"] = crop_size
        random_resized_crop_paras["target_width"] = crop_size
        if "scale" not in random_resized_crop_paras:
            random_resized_crop_paras["scale"] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS["scale"]
        if "aspect_ratio" not in random_resized_crop_paras:
            random_resized_crop_paras["aspect_ratio"] = _RANDOM_RESIZED_CROP_DEFAULT_PARAS["aspect_ratio"]

    rgb_transform = Compose(
        (
            _get_augmentation(aug_type=aug_type, aug_paras=aug_paras) if mode == "train" else []
        )
        + ([ConvertUint8ToFloat()] if convert_to_float else [])
        + [Normalize(mean=video_mean, std=video_std)])
    
    geo_transform = Compose(
            (
                ([RandomResizedCrop(**random_resized_crop_paras)]
                if random_resized_crop_paras is not None
                else [
                    RandomShortSideScale(
                        min_size=min_size,
                        max_size=max_size,
                    ),
                    RandomCrop(size=crop_size),
                ]
                + [RandomHorizontalFlip(p=horizontal_flip_prob)]
            )
            if mode == "train"
            else [
                ShortSideScale(size=min_size),
                CenterCrop(size=crop_size),
            ])
            +([] if (format == "CTHW") else [Permute((1, 0, 2, 3))])
        )
    return rgb_transform, geo_transform