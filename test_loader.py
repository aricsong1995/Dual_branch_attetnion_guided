#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28
# --- Import  --- #
import os
from PIL import Image
import torch.utils.data as data
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import torchvision.utils as utils

# --- Validation/test dataset --- #
class ValLoader(data.Dataset):
    def __init__(self, root_dir, infrence_only=False, crop_size="raw", category="teacher"):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.category = category
        self.infrence_only = infrence_only
        if not infrence_only:
            txt_list_path = os.path.join(root_dir, "val.txt").replace("\\", "/")
        else:
            txt_list_path = os.path.join(root_dir, "student_val.txt").replace("\\", "/")

        with open(txt_list_path, "r") as f:
            testlist = f.readlines()
            testlist = [item.strip() for item in testlist]
        self.test_list = testlist

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, index):
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self.infrence_only:
            haze = Image.open(os.path.join(self.root_dir, "validation", self.test_list[index]).replace("\\", "/"))
            gt = Image.open(os.path.join(self.root_dir, "validation", self.test_list[index]).replace("\\", "/"))
        else:
            haze = Image.open(os.path.join(self.root_dir, "HAZY", self.test_list[index]).replace("\\", "/"))
            gt = Image.open(os.path.join(self.root_dir, "GT", self.test_list[index]).replace("\\", "/"))

        w, h = gt.size

        if self.crop_size != "raw":
            crop_hight, crop_width = self.crop_size
            if crop_hight <= h and crop_width <= w:

                x, y = randrange(w - crop_width + 1), randrange(h - crop_hight + 1)
                haze = haze.crop((x, y, x + crop_hight, y + crop_width))
                gt = gt.crop((x, y, x + crop_width, y + crop_hight))
            else:
                raise ("Bad image size ")

        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze)
        gt = transform_gt(gt)

        img_name = self.test_list[index]
        return haze, gt, img_name


