
from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import msssim
import torchvision.transforms.functional as TF

import torchvision.transforms as transforms
from models.guided_filter import GuidedFilter
from models.fusion_strategy import attention_fusion_weight

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class L_Grad2(nn.Module):
    def __init__(self):
        super(L_Grad2, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_fused = self.sobelconv(image_fused)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_A)
        return Loss_gradient
        
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        #print(image_A.shape)
        #print(image_B.shape)
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)#mse_loss
        #Loss_intensity = F.mse_loss(image_fused, intensity_joint)
        return Loss_intensity

class L_Intensity_color(nn.Module):
    def __init__(self):
        super(L_Intensity_color, self).__init__()

        self.L_color = L_color()
    def forward(self, image_A, image_B, image_fused):
        #print(image_A.shape)
        #print(image_B.shape)
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = torch.mean(angle(image_fused[:, 0, :, :], intensity_joint[:, 0, :, :]) + angle(image_fused[:, 1, :, :], intensity_joint[:, 1, :, :]) + angle(image_fused[:, 2, :, :], intensity_joint[:, 2, :, :]))
        #Loss_intensity = F.l1_loss(self.L_color(image_fused), self.L_color(intensity_joint))#mse_loss
        #Loss_intensity = F.mse_loss(image_fused, intensity_joint)
        return Loss_intensity

def gf_loss(IR,VIS):
    r = 4
    eps = 0.05
    s = 1
    IR_smoothed = GuidedFilter(r, eps)(IR, IR)
    VIS_smoothed = GuidedFilter(r, eps)(VIS, VIS)
    IR_detail = IR - IR_smoothed
    r = 4
    eps = 0.05 * 0.005
    s = 1
    IR_smoothed = GuidedFilter(r, eps)(IR_detail, IR_detail)
    VIS_detail = VIS - VIS_smoothed
    VIS_detail = GuidedFilter(r, eps)(VIS_detail, VIS_detail)

    fusion_out = attention_fusion_weight(IR_smoothed, VIS_detail)
    return fusion_out

def gf_out(output):
    r = 8
    eps =0.05
    s = 1
    output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))

    output_smoothed = GuidedFilter(r, eps)(output, output)
    output_detail = output - output_smoothed

    return output_detail

def angle(a, b):
    vector = torch.mul(a, b)
    up = torch.sum(vector)
    down = torch.sqrt(torch.sum(torch.square(a))) * torch.sqrt(torch.sum(torch.square(b)))
    theta = torch.acos(up / down)  # 弧度制
    return theta


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Grad2 = L_Grad2()
        self.L_Inten = L_Intensity()
        self.L_Intensity_color = L_Intensity_color()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):

        loss_l1 = 1 * (self.L_Inten(image_A, image_B, image_fused))
        loss_gradient = 1 * (self.L_Grad(image_A, image_B, image_fused))
        loss_SSIM = 50 * ( 1 - msssim(gf_loss(image_A, image_B), gf_out(image_fused)))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

