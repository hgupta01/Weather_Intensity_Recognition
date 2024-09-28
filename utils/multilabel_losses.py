import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherDetMultiLabelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        #[N, 7] -> [N, 7]
        torch._assert(input.size() == target.size(), 'Size of input and out tensor do not match')
        return F.binary_cross_entropy_with_logits(input, target.float(), **kwargs)


class WeatherDetMultiClassLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        xrain, xfog, xsnow = input
        loss_rain = F.cross_entropy(xrain, target[:,0])
        loss_fog = F.cross_entropy(xfog, target[:,1])
        loss_snow = F.cross_entropy(xsnow, target[:,2])
        loss_intensity = loss_rain + loss_fog + loss_snow
        return loss_intensity


class WeatherDetMultiClassMultiLabelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        input_dim = input.shape # [N, 9]
        target_dim = target.shape   # [N, 3]

        #[N, 9] -> [N, 3, 3]
        if input.size() != target.size():
            input = input.view(input_dim[0], -1, target_dim[-1])

        loss_intensity = F.cross_entropy(input, target, reduction='none', **kwargs).mean(dim=0).sum()

        target_multilabel = torch.zeros(target.shape, device=input.device, dtype=input.dtype)
        target_multilabel[target>0] = 1.0
        
        input_multilabel = input.mean(-1)
        loss_multiweather = F.binary_cross_entropy_with_logits(input_multilabel, target_multilabel)

        return 0.7*loss_intensity + 0.3*loss_multiweather