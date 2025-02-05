import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision import models
import os,cv2
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
from math import exp
from torchvision.models.vgg import vgg19

# 5.SSIM loss
# 生成一位高斯权重，并将其归一化
def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss/torch.sum(gauss)  # 归一化

# x=gaussian(3,1.5)
# # print(x)
# x=x.unsqueeze(1)
# print(x.shape) #torch.Size([3,1])
# print(x.t().unsqueeze(0).unsqueeze(0).shape) # torch.Size([1,1,1, 3])

# 生成滑动窗口权重，创建高斯核：通过一维高斯向量进行矩阵乘法得到
def create_window(window_size,channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # window_size,1
    # mm:矩阵乘法 t:转置矩阵 ->1,1,window_size,_window_size
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # expand:扩大张量的尺寸，比如3,1->3,4则意味将输入张量的列复制四份，
    # 1,1,window_size,_window_size->channel,1,window_size,_window_size
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# 构造损失函数用于网络训练或者普通计算SSIM值
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# 普通计算SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

#Charrbonnier_Loss
class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x,y):
        diff = torch.add(x, -y)
        value=torch.sqrt(torch.pow(diff,2)+self.epsilon2)
        return torch.mean(value)

#Perceptual_loss vgg19-13
class Vgg19_out(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19_out, self).__init__()
        vgg = models.vgg19(pretrained=False).to(device) #.cuda()
        vgg.load_state_dict(torch.load('/mnt/e/vgg/vgg19-dcbb9e9d.pth'))
        vgg.eval()
        vgg_pretrained_features = vgg.features
        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4): #(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 32):#(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False
 
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class Perceptual_loss13(nn.Module):
    def __init__(self):
        super(Perceptual_loss13, self).__init__()
        self.vgg = Vgg19_out().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss =  self.weights[0]* self.criterion(x_vgg[0], y_vgg[0].detach())+self.weights[2]* self.criterion(x_vgg[2], y_vgg[2].detach())
        return loss
    
class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        loss_network3 = nn.Sequential(*list(vgg.features)[:14]).eval()
        for param in loss_network1.parameters():
            param.requires_grad = False
        for param in loss_network3.parameters():
            param.requires_grad = False
        self.loss_network1 = loss_network1
        self.loss_network3 = loss_network3
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network1(out_images), self.loss_network1(target_images)) + \
                            self.mse_loss(self.loss_network3(out_images), self.loss_network3(target_images))
        return perception_loss
