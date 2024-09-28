from typing import Callable, List, Optional, Type, Union, OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

class TemporalShift(nn.Module):
    """Temporal shift module.

    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_

    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, net, num_segments=3, shift_div=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return self.net(x)

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

def create_tsm_resnet50(num_segments:int=16):
    resnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    n_round = 1
    shift_div = 8
    num_segment_list = [num_segments]*4
    if len(list(resnet.layer3.children())) >= 23:
        n_round = 2

    def make_block_temporal(stage, num_segments):
        """Make temporal shift on some blocks.

        Args:
            stage (nn.Module): Model layers to be shifted.
            num_segments (int): Number of frame segments.

        Returns:
            nn.Module: The shifted blocks.
        """
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(
                    b.conv1,
                    num_segments=num_segments,
                    shift_div=shift_div)
        return nn.Sequential(*blocks)

    resnet.layer1 = make_block_temporal(resnet.layer1, num_segment_list[0])
    resnet.layer2 = make_block_temporal(resnet.layer2, num_segment_list[1])
    resnet.layer3 = make_block_temporal(resnet.layer3, num_segment_list[2])
    resnet.layer4 = make_block_temporal(resnet.layer4, num_segment_list[3])

    tsm_state_dict = torch.load("pretrained/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb_20220831-042b1748.pth", map_location="cpu")['state_dict']
    tsm_state_dict_updated = {}
    for key, value in tsm_state_dict.items():
        if 'backbone' in key:
            key = key.replace('backbone.', '')
            ks = key.split('.')
            if 'conv' in ks[0]:
                ks[1] = ks[1]+ks[0].strip('conv')
                tsm_state_dict_updated['.'.join(ks[1:])] = value
                
            elif 'layer' in ks[0]:
                if 'conv' in ks[2]:
                    ks[3] = ks[3]+ks[2].strip('conv')
                    tsm_state_dict_updated['.'.join(ks[:2]+ks[3:])] = value
                elif ks[2]=='downsample':
                    ks[3] = '0' if ks[3]=='conv' else '1'
                    tsm_state_dict_updated['.'.join(ks)] = value
    tsm_state_dict_updated = OrderedDict(tsm_state_dict_updated)
    resnet.load_state_dict(tsm_state_dict_updated)
    return resnet


def create_tsm_flow_resnet50(num_segments:int=16):
    resnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])
    setattr(resnet, 'conv1', torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False))
    tsm_state_dict = torch.load('pretrained/tsm_kinetics_flow_resnet50.pt', map_location="cpu")
    resnet.load_state_dict(tsm_state_dict)
    
    n_round = 1
    shift_div = 8
    num_segment_list = [num_segments]*4
    if len(list(resnet.layer3.children())) >= 23:
        n_round = 2

    def make_block_temporal(stage, num_segments):
        """Make temporal shift on some blocks.

        Args:
            stage (nn.Module): Model layers to be shifted.
            num_segments (int): Number of frame segments.

        Returns:
            nn.Module: The shifted blocks.
        """
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(
                    b.conv1,
                    num_segments=num_segments,
                    shift_div=shift_div)
        return nn.Sequential(*blocks)

    resnet.layer1 = make_block_temporal(resnet.layer1, num_segment_list[0])
    resnet.layer2 = make_block_temporal(resnet.layer2, num_segment_list[1])
    resnet.layer3 = make_block_temporal(resnet.layer3, num_segment_list[2])
    resnet.layer4 = make_block_temporal(resnet.layer4, num_segment_list[3])
    return resnet