#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28

# --- Import --- #

import os
import torch
import torch.nn as nn
from model.Dehaze import Dehaze
import torch.nn.functional as F
from trainConfig import testConfig
from test_loader import ValLoader
from torch.utils.data import DataLoader
from utils import student_validation


best_weight = '/content/drive/MyDrive/NTIRE2021/DATA_Code/best_weight/student_epoch_15.pth'
# best_weight='saved_checkpoints/student/21_02_2021_19_34_03/student_epoch_170.pth'
infrence_only = False
data_dir = "./data/train/"

# --- Test --- #
def test():
    # --- Gpu device --- #
    device_ids = [0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Devide set to {} ,Device Id set to {}".format(device, device_ids))
    print("Student Inference Mode")

    # --- Define the network --- #
    student = Dehaze()
    student.to(device)

    # --- Multi-GPU --- #
    student = nn.DataParallel(student, device_ids=device_ids)
    # --- Load Weight --- #
    try:
        student.load_state_dict(torch.load(best_weight, map_location=device)["model_state"])
        print("Best weight path : {},  Best student weight loaded".format(best_weight))
    except:
        raise ("No student weight loaded")

    val_loader = DataLoader(
        ValLoader(data_dir, infrence_only=infrence_only, crop_size=testConfig.val_crop_size,
                  category=testConfig.category)
        , batch_size=testConfig.val_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    print("Total validation sample is {}".format(len(val_loader)))

    # --- Use the evaluation model in testing --- #
    psnr, ssim = student_validation(student, val_loader, device, category=testConfig.category, save_tag=True,
                                    inference_only=infrence_only)
    print('PSNR: {:.4f},SSIM: {:.4f}'.format(psnr, ssim))


if __name__ == "__main__":
    test()