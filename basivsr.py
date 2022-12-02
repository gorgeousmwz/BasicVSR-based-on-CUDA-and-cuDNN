# This Python file uses the following encoding: utf-8
#https://github.com/sunny2109/BasicVSR_IconVSR_PyTorch/blob/67ea588ecc552a220580aae4b0cd911525a15d5e/code/basicsr/models/archs/basicvsr_arch.py
from torchvision import transforms
import torch
import torch.nn.functional as F 
from torch import nn 
from PIL import Image
import os
import cv2
import time
import glob
from memory_profiler import profile

#from einops import rearrange
from SpyNet import SpyNet
from arch_util import ResidualBlockNoBN, flow_warp, make_layer 

class BasicVSR(nn.Module):
    """BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond
    Only support x4 upsampling.
    Args:
        num_feat (int): Channel number of intermediate features. 
            Default: 64.
        num_block (int): Block number of residual blocks in each propagation branch.
            Default: 30.
        spynet_path (str): The path of Pre-trained SPyNet model.
            Default: None.
    """
    def __init__(self, num_feat=64, num_block=30, spynet_path=None):
        super(BasicVSR, self).__init__()
        self.num_feat = num_feat#中间特征通道数
        self.num_block = num_block#残差块数目

        # Flow-based Feature Alignment 基于光流的特征对齐模块
        self.spynet = SpyNet(load_path=spynet_path)

        # Bidirectional Propagation 双向传播模块
        self.forward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)#前向
        self.backward_resblocks = ConvResBlock(num_feat + 3, num_feat, num_block)#后向

        # Concatenate Aggregation 融合模块
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=1, stride=1, padding=0, bias=True)

        # Pixel-Shuffle Upsampling 亚像素上采样模块
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=2)
        self.up2 = PSUpsample(num_feat, 64, scale_factor=2)

        # The channel of the tail layers is 64
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # Global Residual Learning 残差学习
        self.img_up = nn.Upsample(scale_factor=4, mode='nearest')

        # Activation Function 激活层
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs):#计算前向后向分支光流
        """Compute optical flow using SPyNet for feature warping.
        Args:
            lrs (tensor): LR frames, the shape is (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 
            forward_flow refers to the flow from current frame to the previous frame. 
            backward_flow is the flow from current frame to the next frame.
        """
        #获取前后向分支
        n, t, c, h, w = lrs.size()
        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)    # n t c h w -> (n t) c h w
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)  # n t c h w -> (n t) c h w
        #光流估计
        forward_flow = self.spynet(forward_lrs, backward_lrs).view(n, t-1, 2, h, w)
        backward_flow = self.spynet(backward_lrs, forward_lrs).view(n, t-1, 2, h, w)

        return forward_flow, backward_flow

    def forward(self, lrs):
        n, t, c, h, w = lrs.size()
        time_consming=[]
        assert h >= 64 and w >= 64, (#输入限制
            'The height and width of input should be at least 64, but got {h} and {w}.'.format(h,w))
        start=time.time()
        #获取光流
        forward_flow, backward_flow = self.comp_flow(lrs)
        end=time.time()
        time_consming.append(end-start)

        # forward_flow = rearrange(forward_flow, 'n t c h w -> t n h w c').contiguous()
        # backward_flow = rearrange(backward_flow, 'n t c h w -> t n h w c').contiguous()
        # lrs = rearrange(lrs, 'n t c h w -> t n c h w').contiguous()

        # Backward Propagation 后向分支
        start=time.time()
        rlt = []#后向对齐特征序列
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)#传播特征
        for i in range(t-1, -1, -1):#逆向遍历每一帧
            curr_lr = lrs[:, i, :, :, :]#当前帧
            if i < t-1:
                flow = backward_flow[:, i, :, :, :]# flow estimation module（获取下一帧的光流）
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))# spatial warping module
            
            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)#与当前帧融合
            feat_prop = self.backward_resblocks(feat_prop)# residual blocks
            rlt.append(feat_prop)#加入特征序列
        rlt = rlt[::-1]#倒序
        end=time.time()
        time_consming.append(end-start)

        # Forward Propagation 前向分支
        start=time.time()
        feat_prop = torch.zeros_like(feat_prop)#传播特征
        for i in range(0, t):#正向遍历每一帧
            curr_lr = lrs[:, i, :, :, :]#当前帧
            if i > 0:
                flow = forward_flow[:, i-1, :, :, :]# flow estimation module（获取上一帧的光流）
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))# spatial warping module
            
            feat_prop = torch.cat([curr_lr, feat_prop], dim=1)#与当前帧融合
            feat_prop = self.forward_resblocks(feat_prop)# residual blocks

            # Fusion and Upsampling
            ##前后向分支对齐特征融合
            cat_feat = torch.cat([rlt[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            #上采样
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            # Global Residual Learning
            base = self.img_up(curr_lr)

            sr_rlt += base
            rlt[i] = sr_rlt
        end=time.time()
        time_consming.append(end-start)

        return torch.stack(rlt, dim=1),time_consming

# Conv + ResBlock
class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        '''
        in_feat:输入通道数
        out_feat:输出通道数
        num_block:残差块数
        '''
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        conv_resblock.append(nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x.to(torch.float32))

# Upsampling with Pixel-Shuffle
class PSUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        '''
        in_feat:输入通道数
        out_feat:输出通道数
        scale_factor:上采样因子
        '''
        super(PSUpsample, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat*scale_factor*scale_factor, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        #[N,C,W,H]->[N,C*r*r,W,H]
        x = self.up_conv(x)
        #周期筛选：[N,C*r*r,W,H]->[N,C,W*r,H*r]
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)

