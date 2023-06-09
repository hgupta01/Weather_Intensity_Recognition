import math
import numbers
import warnings
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

import einops

import torch
from torch import Tensor
from pytorchvideo.transforms.functional import _interpolate_opencv
import torchvision.transforms.functional as F


class Rearrange(object):
    """Rearrange input format required output format using einops.

    Args:
        desired_format: format in einops ('b h w c -> b c h w')
    """
    def __init__(self, desired_format:str):
        self.desired_format = desired_format
        
    def __call__(self, tensor: torch.Tensor):
        return einops.rearrange(tensor, self.desired_format)
    

class StackPIL(object):
    """
        Stack the PIL images to (C T H W) format.
    """
    def __init__(self, stack=True):
        self.stack = stack
        
    def __call__(self, img_list: List[Image.Image]):
        imgs_list = [np.array(img)[:,:,None,:] for img in img_list]
        imgs_list = np.concatenate(imgs_list, axis=2) # H W T C
        imgs_tensor = torch.from_numpy(imgs_list).permute(3, 2, 0, 1).contiguous()
        return imgs_tensor
    
class MergeImageStack(object):
    def __init__(self, stack=True):
        self.stack = stack
        
    def __call__(self, tensor: torch.Tensor):
        c, t, h, w = tensor.shape
        return tensor.reshape(c*t, h, w).contiguous()


class RandomErasingVideo(torch.nn.Module):
    """Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.
    Returns:
        Erased Image.
    """

    def __init__(self, p=0.5, scale_area=(0.1, 0.25), scale_volume=(0.05,0.1),ratio=(0.8, 1.2), value=0, inplace=False):
        super().__init__()
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError("Argument value should be either a number or str or a sequence")
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale_area, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(scale_volume, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale_volume[0] > scale_volume[1]) or (ratio[0] > ratio[1]) or (scale_area[0] > scale_area[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale_area[0] < 0 or scale_area[1] > 1:
            raise ValueError("Scale Area should be between 0 and 1")
        if scale_volume[0] < 0 or scale_volume[1] > 1:
            raise ValueError("Scale Area should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale_area = scale_area
        self.scale_volume = scale_volume
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(
                    img: Tensor,
                    scale_area: Tuple[float, float],
                    scale_volume: Tuple[float, float],
                    ratio: Tuple[float, float],
                    value: Optional[List[float]] = None
                  ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image to be erased. [..., T, H, W]
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_t, img_h, img_w = img.shape[-4], img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w
        volume = area*img_t

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale_area[0], scale_area[1]).item()
            erase_volume = volume * torch.empty(1).uniform_(scale_volume[0], scale_volume[1]).item() 
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            t = int(erase_volume/(h*w))
            if not (h < img_h and w < img_w and t < img_t):
                continue

            if value is None:
                v = torch.empty([img_c, img_t, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            k = torch.randint(0, img_t - t + 1, size=(1,)).item()
            return i, j, k, h, w, t, v

        # Return original image
        return 0, 0, 0, img_h, img_w, img_t, img

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image to be erased.
        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    f"{img.shape[-3]} (number of input channels)"
                )

            x, y, z, h, w, t, v = self.get_params(img, scale_area=self.scale_area,scale_volume=self.scale_volume, ratio=self.ratio, value=value)
            
            if not self.inplace:
                img = img.clone()

            img[..., z : z + t, x : x + h, y : y + w] = v
            return img
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}"
            f"(p={self.p}, "
            f"scale_area={self.scale_area}, "
            f"scale_volume={self.scale_volume}, "
            f"ratio={self.ratio}, "
            f"value={self.value}, "
            f"inplace={self.inplace})"
        )
        return 


def scale_short_side_and_aspect_ratio(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
    change_aspect_ratio = True
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        backend (str): backend used to perform interpolation. Options includes
            `pytorch` as default, and `opencv`. Note that opencv and pytorch behave
            differently on linear interpolation on some versions.
            https://discuss.pytorch.org/t/pytorch-linear-interpolation-is-different-from-pil-opencv/71181
    Returns:
        An x-like Tensor with scaled spatial dims.
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ("pytorch", "opencv")
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    if change_aspect_ratio:
        if (float(new_w)/new_h > 1.5):
            new_h = int(math.floor((float(new_w) / 1.3333)))
        elif (float(new_h)/new_w > 1.5):
            new_w = int(math.floor((float(new_h) / 1.3333)))

    if backend == "pytorch":
        return torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=False
        )
    elif backend == "opencv":
        return _interpolate_opencv(x, size=(new_h, new_w), interpolation=interpolation)
    else:
        raise NotImplementedError(f"{backend} backend not supported.")


class RandomScaleShortSideAndAspectRatio(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
        change_aspect_ratio = True,
    ):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend
        self._change_aspect_ratio = change_aspect_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return scale_short_side_and_aspect_ratio(
            x, size, self._interpolation, self._backend, self._change_aspect_ratio
        )


class RandomGaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, p, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = [kernel_size, kernel_size]
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        self.p = p
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        if torch.rand(1) < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        
        return img

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma}, p={self.p})"
        return s


class RandomRotation90(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        k = torch.randint(4, size=(1,1)).item()

        return torch.rot90(img, k=k, dims=[-2, -1])

