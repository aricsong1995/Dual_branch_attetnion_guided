#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28
# --- Import --- #
import os
import time

# --- Training Output Location and Setting --- #
class trainConfig:
    category = "student"

    if category in "student":
        learning_rate = []

        # train_batch_size = 2
        val_batch_size = 1

    data_dir = './data'
    val_crop_size = "raw"

    job_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    save_best = './best_weight'
    if not os.path.exists(save_best):
        os.makedirs(save_best)
    checkpoints = os.path.join('saved_checkpoints', category, job_time)
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)
    txt_log = os.path.join('training_log', category, job_time)
    if not os.path.exists(txt_log):
        os.makedirs(txt_log)
    writer_path = os.path.join('run', category, job_time)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)


# --- Testing Output Location and Setting --- #
class testConfig:
    category = "student"

    if category in "student":
        val_batch_size = 1
        best_student_weight = "KTDN.pth"
        val_crop_size = "raw"
        Infrence_only = True  # set to True when no GT provided

    data_dir = './data'

