# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        if len(self._data[2:])==1:
            return int(self._data[2])
        else:
            return list([int(l) for l in self._data[2:]])

class VideoDataSet(data.Dataset):
    def __init__(self, 
                 root_path, 
                 list_file,
                 num_frames=8,
                 image_tmpl='img_{:03d}.jpg',
                 transform=None,
                 random_shift=True,
                 train_mode=True,
                 remove_missing=False):
        self.root_path = root_path
        self.list_file = list_file
        self.num_frames = num_frames
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.train_mode = train_mode
        self.remove_missing = remove_missing
        self._parse_list()

    def _load_image(self, directory, idx):
        img_path = os.path.join(self.root_path, 'images', directory, self.image_tmpl.format(idx))
        try:
            return [np.asarray(Image.open(img_path).convert('RGB'))]
        except Exception:
            print('error loading image:', img_path)

    def _parse_list(self):
        tmp = [x.strip().split(" ") for x in open(os.path.join(self.root_path, 'labels', self.list_file))]
        if self.remove_missing:
            tmp = [item for item in tmp if ((int(item[1]) >= 3) and os.path.exists(os.path.join(self.root_path, 'images', item[0])))]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if record.num_frames > self.num_frames:
            offsets = np.sort(random.sample(list(range(record.num_frames)), self.num_frames))
        else:
            offsets = np.array(list(range(record.num_frames))+[record.num_frames-1]*(self.num_frames-record.num_frames))
        return offsets + 1

    def _get_val_indices(self, record):
        if (record.num_frames) > self.num_frames:
            tick = (record.num_frames) / float(self.num_frames)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_frames)])
        else:
            offsets = np.array(list(range(record.num_frames))+[record.num_frames-1]*(self.num_frames-record.num_frames))
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.train_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_val_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()  # list of numpy image object
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
        images = torch.stack([torch.from_numpy(img.copy()) for img in images]).permute(3, 0, 1, 2) #CTHW
        if self.transform:
            process_data = self.transform(images)
        else:
            process_data = images
        return process_data, np.asarray(record.label)

    def __len__(self):
        return len(self.video_list)


class VideoFlowDataSet(VideoDataSet):
    def __init__(self, 
                 root_path, 
                 list_file,
                 num_frames=3,
                 image_tmpl='img_{:03d}.jpg',
                 flow_tmpl='{}_{:03d}.jpg',
                 rgb_transform=None,
                 geo_transform=None,
                 random_shift=True,
                 train_mode=True,
                 remove_missing=False):
        super().__init__(root_path, 
                         list_file,
                         num_frames=num_frames,
                         image_tmpl=image_tmpl,
                         transform=None,
                         random_shift=random_shift,
                         train_mode=train_mode,
                         remove_missing=remove_missing)
        self.flow_tmpl = flow_tmpl
        self.rgb_transform = rgb_transform
        self.geo_transform = geo_transform
        self.train_mode = train_mode
            
    def _load_flow(self, directory, idx):
        x_path = os.path.join(self.root_path, 'flows', directory, self.flow_tmpl.format('x', idx))
        y_path = os.path.join(self.root_path, 'flows', directory, self.flow_tmpl.format('y', idx))
        try:
            return [np.stack((np.asarray(Image.open(x_path)), np.asarray(Image.open(y_path))))]
        except Exception:
            print('error loading image:', x_path)
            
    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if (record.num_frames-1) > self.num_frames:
            offsets = np.sort(random.sample(list(range(record.num_frames-1)), self.num_frames))
        else:
            offsets = np.array(list(range(record.num_frames-1))+[record.num_frames-2]*(self.num_frames-(record.num_frames-1)))
        return offsets + 1

    def _get_val_indices(self, record):
        if (record.num_frames-1) > self.num_frames:
            tick = (record.num_frames-1) / float(self.num_frames)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_frames)])
        else:
            offsets = np.array(list(range(record.num_frames-1))+[record.num_frames-2]*(self.num_frames-(record.num_frames-1)))
        return offsets + 1

    def get(self, record, indices):
        images = list()  # list of numpy image object
        flows = list()
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)
            flow_img = self._load_flow(record.path, p)
            images.extend(seg_imgs)
            flows.extend(flow_img)
        images = torch.stack([torch.from_numpy(img.copy()) for img in images]).permute(3, 0, 1, 2) #CTHW
        flows = 2*((torch.stack([torch.from_numpy(img.copy()) for img in flows]).permute(1, 0, 2, 3).float() / 255.0)-0.5) #CTHW
        if self.rgb_transform:
            images = self.rgb_transform(images)
        comb = torch.cat((images, flows), dim=0)
        if self.geo_transform:
            comb = self.geo_transform(comb)
        images, flows = torch.split(comb, [3, 2], dim=1)
        return images, flows, np.asarray(record.label)

