#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28

# --- Import  --- #
import torch
import torch.nn as nn
from model.Dehaze import Dehaze
import torch.nn.functional as F
from pytorch_msssim import SSIM
from trainConfig import trainConfig
from train_loader import TrainLoader
from test_loader import ValLoader
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from loss import LossNetwork, LapLoss, GANLoss
from utils import adjust_learning_rate, to_psnr, to_ssim_skimage, print_log, student_validation
import time
from torch.utils.tensorboard import SummaryWriter
from utils import seed_everything, UnNormalize

# --- Set Random Seed  --- #
seed_everything()

# --- Set SSIM Loss  --- #
class SSIM_Loss(SSIM):
    def forward(self, pred, gt):
        return 1 - super(SSIM_Loss, self).forward(pred, gt)

# --- Tensorboard Log--- #
writer = SummaryWriter(trainConfig.writer_path)
print('writer_path : tensorboard --logdir={}'.format(trainConfig.writer_path))


# --- hyper-parameters-- #
train_crop_size = [256, 256]
epoch_number = 20
learing_rate_range = [1e-4, 5e-5]
lambda_l = 0.2
lambda_s = 0.336

# --- Gpu device --- #
device_ids = [0]
checkpoints = '/content/drive/MyDrive/NTIRE2021/DATA_Code/saved_checkpoints/student/25_06_2021_19_17_49/student_epoch_28.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Devide set to {} ,Device Id set to {}".format(device, device_ids))
print("Student Training Mode")

# --- Define the network --- #
student = Dehaze()
student.to(device)


# --- Multi-GPU --- #
student = nn.DataParallel(student, device_ids=device_ids)


# --- Calculate all trainable parameters in network --- #
total_para = sum(para.numel() for para in student.parameters() if para.requires_grad)
print("Total parameter {}".format(total_para))

# --- Load Network-- #
try:
    student.load_state_dict(
        torch.load(checkpoints, map_location=device)["model_state"])
    print("Load checkpoints {}".format(checkpoints))

except:
    print("Only best Teacher weight loaded or No weight Loaded")

# --- Define the perceptual loss network and other losses --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()
loss_s = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

# --- Build Optimizer --- #
optimizer = torch.optim.Adam(student.parameters(), lr=learing_rate_range[0])

# --- Load training data and validation/test data --- #
train_loader = DataLoader(
    TrainLoader('./data/train', crop_size=train_crop_size, category=trainConfig.category)
    , batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(
    ValLoader('./data/train', crop_size=trainConfig.val_crop_size, category=trainConfig.category)
    , batch_size=trainConfig.val_batch_size, shuffle=False, num_workers=4, pin_memory=False)
print("Total train sample is {}, total validation sample is {}".format(len(train_loader), len(val_loader)))

old_psnr, old_ssim = student_validation(student, val_loader, device, category=trainConfig.category, save_tag=True)
print('Previous PSNR: {:.4f}, previous ssim: {:.4f}'.format(old_psnr, old_ssim))
iteration = 0

# --- Train Stage --- #
for epoch in range(epoch_number):
    psnr_list = []
    ssim_list = []
    start_time = time.time()
    epoch_loss = 0
    lr = adjust_learning_rate(optimizer, epoch, epoch_number, writer, category="student",
                              learning_rate=learing_rate_range)
    for batch_id, train_data in enumerate(train_loader):
        haze, gt, img_name = train_data
        haze = haze.to(device)  # -1,1
        gt = gt.to(device)
        student.train()
        dehaze = student(haze)
        loss_l1 = F.l1_loss(dehaze, gt)
        with torch.no_grad():
            loss_percep = loss_network(dehaze, gt)
            loss_ssim = loss_s(dehaze, gt)

        student_loss = loss_l1 + lambda_l * loss_percep + lambda_s * loss_ssim

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        student_loss.backward()
        optimizer.step()
        if batch_id % 40 == 0:
            print("epoch : {}/{}, Iteration : {}, student_Loss:{} ".format(epoch, epoch_number, batch_id,
                                                                           student_loss.item()))
        iteration += 1
        epoch_loss += student_loss.item()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))
        ssim_list.extend((to_ssim_skimage(dehaze, gt)))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)
    train_ssim = sum(ssim_list) / len(ssim_list)
    writer.add_scalars("Loss", {"Epoch_loss": epoch_loss, }, epoch)
    writer.add_scalars("eval/train_matrix_group",
                       {
                           "PSNR_train": train_psnr,
                           "SSIM_train": train_ssim,
                       }, epoch)

    # --- Save the network parameters --- #
    torch.save({"model_state": student.state_dict(), "lr": optimizer.state_dict()['param_groups'][0]['lr']},
               '{}/{}_epoch_{}.pth'.format(trainConfig.checkpoints, trainConfig.category, epoch))
    print("Checkpoint saved")
    one_epoch_time = time.time() - start_time

    # --- Use the evaluation model in testing --- #
    student.eval()
    val_psnr, val_ssim = student_validation(student, val_loader, device, category=trainConfig.category,
                                            save_tag=True, inference_only=False)
    print_log(epoch, epoch_number, one_epoch_time, train_psnr, val_psnr, val_ssim, jobtime=trainConfig.writer_path,
              logdir=trainConfig.txt_log, category=trainConfig.category)
    writer.add_scalars(
        'eval/validation_matrix_group',
        {
            "PSNR_Validation": val_psnr,
            "SSIM_Validation": val_ssim,
        }, epoch)

    # --- update the network weight --- #
    if val_psnr > old_psnr:
        torch.save({"model_state": student.state_dict()},
                   '{}/{}_best.pth'.format(trainConfig.save_best, trainConfig.category))
        old_psnr = val_psnr
        print("Best weight saved")



