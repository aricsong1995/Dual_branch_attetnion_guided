#!/usr/bin/env python
# coding: utf-8
# Auther:Xiang Song
# Project: A Dual-Branch Attention Guided Context Aggregation Network for NonHomogeneous Dehazing
# Modified date: 2021-07-28

# --- Imports --- #
import torch
import torch.nn as nn
import torch.functional as F
import model.Res2Net as Pre_Res2Net
import math
import torch.utils.model_zoo as model_zoo
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DCCL(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DCCL,self).__init__()
        self.conv_d1=nn.Conv2d(inplanes,outplanes,kernel_size=3,padding=1,dilation=1)
        self.conv_d3=nn.Conv2d(inplanes,outplanes,kernel_size=3,padding=3,dilation=3)
        self.conv_d5=nn.Conv2d(inplanes,outplanes,kernel_size=3,padding=5,dilation=5)

        self.fuse=nn.Conv2d(outplanes*3,outplanes,kernel_size=1)
    def forward(self,x):
        out_1=self.conv_d1(x)
        out_2=self.conv_d3(x)
        out_3=self.conv_d5(x)

        out=torch.cat((out_1,out_2,out_3),dim=1)
        out=self.fuse(out)
        return out


class context_residual_block(nn.Module):
    def __init__(self,inplanes=64,outplanes=64,res_unit=16):
        super(context_residual_block,self).__init__()
        self.act=nn.PReLU()
        self.dccl_1=DCCL(inplanes,outplanes)
        self.bn1 =nn.BatchNorm2d(outplanes)
        self.dccl_2=DCCL(outplanes,outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.res_unit=res_unit
    def forward(self,x):
        out=self.dccl_1(x)
        out=self.bn1(out)
        out=self.act(out)

        out = self.dccl_2(out)
        out = self.bn2(out)
        out = self.act(out)

        return x+out

# --- AGCA Branch --- #
class Detail_Repair_Net(nn.Module):
    def __init__(self,inplanes=3,outplanes=64,res_unit=16):
        super(Detail_Repair_Net, self).__init__()
        self.head=nn.Conv2d(inplanes,outplanes,kernel_size=3,padding=1)
        self.act1=nn.PReLU()
        body=[]
        for i in range(res_unit):
            body.append(context_residual_block(outplanes,outplanes,res_unit))
        self.body=nn.Sequential(*body)

        self.conv2=nn.Sequential(nn.Conv2d(outplanes,outplanes,kernel_size=3,padding=1),
                                 nn.BatchNorm2d(outplanes))



        self.tail=nn.Sequential(nn.Conv2d(outplanes,3,kernel_size=3,padding=1),
                                )
    def forward(self,x):
        residual=self.act1(self.head(x))
        out=self.body(residual)
        out=self.conv2(out)


        out=out+residual
        out=self.tail(out)
        # out=(out+1)/2
        return out

# --- Knowledge Transfer Branch --- #
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # out_channel=512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_init = self.relu(x)
        x = self.maxpool(x_init)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_output = self.layer3(x_layer2)
        return x_init, x_layer1, x_layer2, x_output

# --- Attention module --- #
class Attention_module(nn.Module):
    def __init__(self,conv,dim,kernel_size):
        super(Attention_module,self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
class knowledge_adaptation_UNet(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_UNet, self).__init__()
        self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101 = Pre_Res2Net.Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
        res2net101.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s'],map_location=torch.device('cpu')))
        pretrained_dict = res2net101.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

        self.up_block= nn.PixelShuffle(2)
        self.attention0 = Attention_module(default_conv, 1024, 3)
        self.attention1 = Attention_module(default_conv, 256, 3)
        self.attention2 = Attention_module(default_conv, 192, 3)
        self.attention3 = Attention_module(default_conv, 112, 3)
        self.attention4 = Attention_module(default_conv, 44, 3)

        self.conv_process_1 = nn.Conv2d(44, 44, kernel_size=3,padding=1)
        self.conv_process_2 = nn.Conv2d(44, 28, kernel_size=3,padding=1)
    def forward(self, input):

        x_inital, x_layer1, x_layer2, x_output = self.encoder(input)
        x_mid = self.attention0(x_output)
        x = self.up_block(x_mid)
        x = self.attention1(x)
        x = torch.cat((x, x_layer2), 1)
        x = self.up_block(x)
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention3(x)
        x = torch.cat((x, x_inital), 1)
        x = self.up_block(x)
        x = self.attention4(x)
        x = self.conv_process_1(x)
        out = self.conv_process_2(x)
        return out

# --- Main Network --- #
class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()
        self.knowledge_adaptation_branch=knowledge_adaptation_UNet()
        self.detail_repair_net=Detail_Repair_Net(inplanes=3,outplanes=64,res_unit=14)
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(31, 3, kernel_size=7, padding=0), nn.Tanh())
    def forward(self, input):
        out=self.detail_repair_net(input)
        knowledge_adaptation_branch_out=self.knowledge_adaptation_branch(input)
        x = torch.cat([out, knowledge_adaptation_branch_out], 1)
        x = self.fusion(x)
        x=(x+1)/2
        return x



