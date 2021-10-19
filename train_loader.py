#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28

# --- Import --- #
import os
from PIL import Image
import torch.utils.data as data
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import torchvision.utils as utils

# --- Random Image Rotation--- #
def rotate(img, rotate_index):
    '''
    :return: 8 version of rotating image
    '''
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return img.rotate(90)
    if rotate_index == 2:
        return img.rotate(180)
    if rotate_index == 3:
        return img.rotate(270)
    if rotate_index == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)

# --- Training dataset --- #
class TrainLoader(data.Dataset):
    def __init__(self, root_dir, crop_size=[512, 512], category="student"):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size

        txt_list_path = os.path.join(root_dir, "train.txt").replace("\\", "/")

        with open(txt_list_path, "r") as f:
            train_list = f.readlines()
            train_list = [item.strip() for item in train_list]
        # print("train_list: {} ".format(train_list))

        self.train_list = train_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # print("haze path {}".format(os.path.join(self.root_dir,"HAZY",self.train_list[index])))
        haze = Image.open(os.path.join(self.root_dir, "HAZY", self.train_list[index]).replace("\\", "/"))
        gt = Image.open(os.path.join(self.root_dir, "GT", self.train_list[index]).replace("\\", "/"))

        if self.crop_size != "raw":
            # data aug
            w, h = gt.size
            crop_hight, crop_width = self.crop_size
            if crop_hight <= h and crop_width <= w:

                x, y = randrange(w - crop_width + 1), randrange(h - crop_hight + 1)
                croped_haze = haze.crop((x, y, x + crop_hight, y + crop_width))
                croped_gt = gt.crop((x, y, x + crop_width, y + crop_hight))
            else:
                raise ("Bad image size ")

        rotate_index = randrange(0, 8)
        croped_haze = rotate(croped_haze, rotate_index)
        croped_gt = rotate(croped_gt, rotate_index)

        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(croped_haze)
        gt = transform_gt(croped_gt)

        img_name = self.train_list[index]
        return haze, gt, img_name

