import torch.utils.data as data
import torch
from PIL import Image
from numpy import array as arr
from numpy import transpose as trans
from numpy import flipud, fliplr, rot90, mean
from os.path import basename, dirname, join, isfile
import numpy as np
import cv2
import random

def MSE(noise, clean):
    return mean((noise - clean) ** 2)

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, crop_size=[128, 128],
                 rotate=True, fliplr=True, fliptb=True):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_dirs = image_dirs
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.crop_w = crop_size[0]
        self.crop_h = crop_size[1]

    def load(self, noise_fn):
        base_path = dirname(dirname(noise_fn))
        clean_fn = join(base_path, 'Clean', basename(noise_fn))

        noise = arr(Image.open(noise_fn))
        clean = arr(Image.open(clean_fn))

        h, w = noise.shape[:2]

        xp = random.randint(0, w-self.crop_w)
        yp = random.randint(0, h-self.crop_h)

        noise = noise[yp:yp + self.crop_h, xp:xp + self.crop_w]
        clean = clean[yp:yp + self.crop_h, xp:xp + self.crop_w]
        return noise, clean

    def __getitem__(self, index):
        noise, clean = self.load(self.image_dirs[index])
        sy_noise, sy_clean = self.load(self.image_dirs[index + 1])

        # random horizontal & vertical flip
        if self.fliplr:
            if random.random() > 0.5:
                noise = fliplr(noise).copy()
                clean = fliplr(clean).copy()
        if self.fliptb:
            if random.random() > 0.5:
                noise = flipud(noise).copy()
                clean = flipud(clean).copy()
        if self.rotate:
            noise = rot90(noise).copy()
            clean = rot90(clean).copy()

        noise = trans(noise, [2, 0, 1])
        clean = trans(clean, [2, 0, 1])
        sy_noise = trans(sy_noise, [2, 0, 1])
        sy_clean = trans(sy_clean, [2, 0, 1])

        e_real = MSE(noise, clean)
        e_syn = MSE(sy_noise, sy_clean)

        if e_real > e_syn:
            order = 0
        else:
            order = 1

        return torch.from_numpy(noise / 255.), \
               torch.from_numpy(sy_noise / 255.), \
               torch.from_numpy(clean / 255.), \
               torch.from_numpy(sy_clean / 255.), \
               order

    def __len__(self):
        return len(self.image_dirs) - 1


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs):
        super(TestDatasetFromFolder, self).__init__()
        self.image_dirs = image_dirs

    def load(self, noise_fn):
        base_path = dirname(dirname(noise_fn))
        clean_fn = join(base_path, 'Clean', basename(noise_fn))

        noise = arr(Image.open(noise_fn))
        clean = arr(Image.open(clean_fn))

        return noise, clean

    def __getitem__(self, index):
        noise, clean = self.load(self.image_dirs[index])

        h, w = noise.shape[:2]
        
        noise = trans(noise, [2, 0, 1])
        clean = trans(clean, [2, 0, 1])


        return torch.from_numpy(noise / 255.), \
               torch.from_numpy(clean / 255.)

    def __len__(self):
        return len(self.image_dirs)
