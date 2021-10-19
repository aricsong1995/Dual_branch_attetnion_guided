#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28

# --- Imports --- #
import time
import torch.nn.functional as F
from math import log10
from skimage import measure
import torchvision.utils as utils
import datetime
import random, torch, os, numpy as np


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                # print(param)
                param.requires_grad = requires_grad


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(dehaze_list))]

    return ssim_list


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # t=t.mul(s)
            # t=t.add(m)
        return tensor


def save_image(dehaze, image_name, category):
    dehaze = dehaze
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    if category not in "slice":

        if not os.path.exists("{}_results".format(category)):
            os.mkdir("{}_results".format(category))

        for ind in range(batch_num):
            utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))

    else:  # sling mode
        sliced_infrence_path = "data/slicing/sliced_inference"
        if not os.path.exists(sliced_infrence_path):
            os.makedirs(sliced_infrence_path)
        for ind in range(batch_num):
            utils.save_image(dehaze_images[ind], '{}/{}'.format(sliced_infrence_path, image_name))
        return sliced_infrence_path


def teacher_validation(net, val_loader, device, category, save_tag=False, inference_only=False):
    psnr = []
    ssim = []
    unorm = UnNormalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    net.eval()
    with torch.no_grad():
        for batch_id, val_data in enumerate(val_loader):
            haze, gt, image_name = val_data
            # haze=haze.to(device)
            gt = gt.to(device)

            rec_clear = net(gt)

            # Todo:unorm
            if inference_only:  # without gt pair
                if save_tag:
                    save_image(rec_clear, image_name, category)
                    psnr = [0]
                    ssim = [0]

            else:
                psnr.extend(to_psnr(rec_clear, gt))
                ssim.extend(to_ssim_skimage(rec_clear, gt))
                save_image(rec_clear, image_name, category)

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)
    return avg_psnr, avg_ssim


def student_validation(net, val_loader, device, category, save_tag=False, inference_only=False):
    psnr = []
    ssim = []
    net.eval()
    all_run_time = 0

    with torch.no_grad():
        start_time = time.time()
        for batch_id, val_data in enumerate(val_loader):
            haze, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            torch.cuda.synchronize()
            start_time = datetime.datetime.now()

            dehaze = net(haze)

            torch.cuda.synchronize()
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).microseconds
            all_run_time += run_time / 1000000.0
            # dehaze=dehaze.clamp(0,1)
            # dehaze=unorm(dehaze).to(device)

            if inference_only:  # without gt pair
                print(image_name)
                if save_tag:
                    save_image(dehaze, image_name, category)
                    psnr = [0]
                    ssim = [0]

            else:
                psnr.extend(to_psnr(dehaze, gt))
                ssim.extend(to_ssim_skimage(dehaze, gt))
                save_image(dehaze, image_name, category)

    print('run time per image: {}'.format(all_run_time / (len(val_loader))))
    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)
    return avg_psnr, avg_ssim

# --- Decay learning rate --- #
def adjust_learning_rate(optimizer, epoch, num_epoch, writer, category="teacher",
                         learning_rate=[2e-4, 2e-4, 1e-4, 5e-5, 2e-5]):
    step = num_epoch // len(learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate[epoch // step]
        print("Learning rate sets to {}, adjust_learning_rate every {} epoch".format(param_group['lr'], step))
        writer.add_scalars("lr/{}_train_lr_group".format(category),
                           {
                               'lr': param_group['lr']
                           }
                           , epoch)
    return param_group['lr']


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, jobtime, logdir, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    print('Training Log Dir : {}'.format(logdir))
    with open('{}/{}_log.txt'.format(logdir, category), 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, JobTimeFolder: {2}s, Epoch: [{3}/{4}], Train_PSNR: {5:.2f}, Val_PSNR: {6:.2f}, Val_SSIM: {7:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),  # 0
                    one_epoch_time, jobtime, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)
        # 1            #2


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
