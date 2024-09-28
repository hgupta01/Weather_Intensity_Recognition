import torch
from . import VideoDataSet, VideoFlowDataSet, create_video_transform, create_video_optflow_transform

def get_train_transform(aug_type:str="randaug", format:str="CTHW", mode: str='rgb'):
    aug_paras = dict(magnitude=5, num_layers=3, prob=0.1)
    
    if mode=='rgb':
        transform = create_video_transform(mode = "train",
                                            convert_to_float = True,
                                            video_mean = (0.45, 0.45, 0.45),
                                            video_std = (0.225, 0.225, 0.225),
                                            min_size = 256,
                                            max_size = 320,
                                            crop_size = 224,
                                            horizontal_flip_prob = 0.5,
                                            aug_type = aug_type, #'default'
                                            aug_paras = aug_paras,
                                            random_resized_crop = False,
                                            format=format)
    elif mode=='flow':
        transform = create_video_optflow_transform(mode = "train",
                                                   convert_to_float = True,
                                                   video_mean = (0.45, 0.45, 0.45),
                                                   video_std = (0.225, 0.225, 0.225),
                                                   min_size = 256,
                                                   max_size = 320,
                                                   crop_size = 224,
                                                   horizontal_flip_prob = 0.5,
                                                   aug_type = aug_type, #'default'
                                                   aug_paras = aug_paras,
                                                   random_resized_crop = False,
                                                   format=format)
    return transform

def get_val_transform(format:str="CTHW", mode: str='rgb'):
    if mode=='rgb':
        transform = create_video_transform(mode = "val",
                                        convert_to_float = True,
                                        video_mean = (0.45, 0.45, 0.45),
                                        video_std = (0.225, 0.225, 0.225),
                                        min_size = 256,
                                        max_size = 320,
                                        crop_size = 224,
                                        horizontal_flip_prob = 0.5,
                                        aug_type = 'default',
                                        aug_paras = None,
                                        random_resized_crop = False,
                                        format=format)
    elif mode == 'flow':
        transform = create_video_optflow_transform(mode = "val",
                                       convert_to_float = True,
                                       video_mean = (0.45, 0.45, 0.45),
                                       video_std = (0.225, 0.225, 0.225),
                                       min_size = 256,
                                       max_size = 320,
                                       crop_size = 224,
                                       horizontal_flip_prob = 0.5,
                                       aug_type = 'default',
                                       aug_paras = None,
                                       random_resized_crop = False,
                                       format=format)
    return transform

def get_test_transform(format:str="CTHW", mode='rgb'):
    if mode=='rgb':
        transform = create_video_transform(mode = "val",
                                        convert_to_float = True,
                                        video_mean = (0.45, 0.45, 0.45),
                                        video_std = (0.225, 0.225, 0.225),
                                        min_size = 224,
                                        crop_size = 224,
                                        horizontal_flip_prob = 0.5,
                                        aug_type = 'default',
                                        aug_paras = None,
                                        random_resized_crop = False,
                                        format=format)
    elif mode=='flow':
        transform = create_video_optflow_transform(mode = "val",
                                        convert_to_float = True,
                                        video_mean = (0.45, 0.45, 0.45),
                                        video_std = (0.225, 0.225, 0.225),
                                        min_size = 224,
                                        crop_size = 224,
                                        horizontal_flip_prob = 0.5,
                                        aug_type = 'default',
                                        aug_paras = None,
                                        random_resized_crop = False,
                                        format=format)
    return transform

def create_video_dataloader(root_path:str, 
                            data_list:str, 
                            batch_size:int=16, 
                            num_frames:int=16, 
                            prefix:str='{:03d}.jpg', 
                            is_train:bool = False, 
                            num_worker:int=0,
                            aug_type:str="default",
                            format:str="CTHW"):
    if is_train:
        transform = get_train_transform(aug_type, format)
    else:
        transform = get_test_transform(format=format)

    data_loader = torch.utils.data.DataLoader(
        VideoDataSet(root_path, 
                     data_list, 
                     num_frames=num_frames,
                     image_tmpl=prefix,
                     transform=transform,
                     train_mode=is_train
                     ),
        batch_size=batch_size, shuffle=is_train,
        num_workers=num_worker, pin_memory=num_worker>0,
        drop_last=is_train)  # prevent something not % n_GPU
    return data_loader


def create_video_opt_dataloader(root_path:str, 
                                data_list:str, 
                                batch_size:int=16, 
                                num_frames:int=16, 
                                prefix:str='{:03d}.jpg', 
                                flow_prefix:str='{}_{:03d}.jpg', 
                                is_train:bool = False, 
                                num_worker:int=0,
                                aug_type:str="default",
                                format:str="TCHW"):
    if is_train:
        transform = get_train_transform(aug_type, format, mode='flow')
    else:
        transform = get_val_transform(format=format, mode='flow')

    data_loader = torch.utils.data.DataLoader(
        VideoFlowDataSet(root_path, 
                        data_list, 
                        num_frames=num_frames,
                        image_tmpl=prefix,
                        flow_tmpl=flow_prefix,
                        rgb_transform=transform[0],
                        geo_transform=transform[1],
                        train_mode=is_train
                        ),
        batch_size=batch_size, shuffle=is_train,
        num_workers=num_worker, pin_memory=num_worker>0,
        drop_last=is_train)  # prevent something not % n_GPU
    return data_loader