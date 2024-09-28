import torch
from models.swin_transformer import create_swin_transformer
from models.tsm_resnet import create_tsm_resnet50, create_tsm_flow_resnet50
from models.timesformer.mvit_v2 import create_mvitv2

def create_backbone(name:str)->torch.nn.Module:
    if name=="x3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", model="x3d_m", pretrained=True)
        delattr(model.blocks, str(len(model.blocks)-1))
        return model, 192
    elif name=="i3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", model="i3d_r50", pretrained=True)
        delattr(model.blocks, str(len(model.blocks)-1))
        return model, 2048
    elif name=="swin":
        return create_swin_transformer(), 768
    elif name=="tsm":
        return create_tsm_resnet50(), 2048
    elif name=="mvit":
        return create_mvitv2(), 768
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_opt_backbone()->torch.nn.Module:
    return create_tsm_flow_resnet50()