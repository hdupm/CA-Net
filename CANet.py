from telnetlib import X3PAD
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from torch.distributions import kl
import torch.nn.functional as F
from loss import CharbonnierLoss, Perceptual_loss13, SSIM
import functools
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

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
    
class AFF(nn.Module):
    def __init__(self, channels=16, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual,rcb):
        xa = x + residual+rcb
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = x * wei + residual * (1 - wei)
        return xo

class CINR(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CINR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1,padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channel, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        return y
      
class CINR_3D(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CINR_3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 3, 3), stride=1,padding=(0, 1, 1))
        self.norm1 = nn.InstanceNorm3d(out_channel, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        return y
    
class CINR_PW(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(CINR_PW, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channel, affine=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x
    
class Contextattention(nn.Module):
    def __init__(self, channels=16, r=4,bias=True):
        super(Contextattention, self).__init__()
        self.conv_key = nn.Conv2d(channels, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.act=nn.ReLU(inplace=True)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        )
        inter_channels = int(channels // r)
        #local branch
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def golbalbranch(self, x):
        b, c, h, w = x.size()
        query = x
        query = query.view(b, c, h * w)#query
        # query = query.unsqueeze(1)#(b,1,c,h*w)
        # key = self.conv_key(x)#(b,1,h,w)
        # key = key.view(b, 1, h * w)
        # key = self.softmax(key)#(b,1,h*w)
        # key = key.unsqueeze(3)#(b,1,h*w,1)
        key = self.softmax(self.conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())#(b,h*w,1)
        context = torch.matmul(query, key)#(b,1,c,1)
        context = context.view(b, c, 1, 1)
        globalcontext = self.transform(context)        
        return globalcontext

    def forward(self, x):
        globalcontext = self.golbalbranch(x)
        local_att=self.local_att(x)
        global_att=self.global_att(x)
        localcontext=global_att+local_att                
        output = self.act(localcontext + globalcontext)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y) 


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DGM(nn.Module):
    def __init__(self, lowchannels,highchannels):
        super(DGM, self).__init__()                
        self.conv3x3 = nn.Conv2d(highchannels, highchannels, kernel_size=3, stride=1,padding=1, bias=False)
        self.bnhigh = nn.BatchNorm2d(highchannels)
        self.conv1x1 = nn.Conv2d(lowchannels, highchannels, kernel_size=1, padding=0, bias=False)
        self.bnlow = nn.BatchNorm2d(lowchannels)    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, low,high):
        b, c, h, w = low.shape#b,c,h,w
        #获取低级特征的信息并进行处理
        lowfeatures = nn.AvgPool2d(low.shape[2:])(low).view(len(low), c, 1, 1)#
        lowfeatures = self.conv1x1(lowfeatures)
        lowfeatures = self.bnlow(lowfeatures)
        lowfeatures = self.relu(lowfeatures)
        
        #对高级特征进行处理
        highfeatures = self.conv3x3(high)
        highfeatures = self.bnhigh(highfeatures)        

        #低级特征对高级特征进行加权
        attention = highfeatures * lowfeatures
        
        #高级信息加上加权特征
        out = self.relu(high + attention)
        return out

class DRM(nn.Module):
    def __init__(self):
        super(DRM, self).__init__()
        self.conv_in = nn.Conv2d(3, 16, 1, 1, bias=True) 
        self.conv1 = CINR(16,64)     
        self.conv2 = CINR(64,64)
        self.max1=nn.MaxPool2d(2,2)       
        self.conv3 = CINR(64,128)  
        self.max2=nn.MaxPool2d(2,2) 
        self.conv4 = CINR(128,256) 
        self.res1=ResidualBlock(256)
        self.res2=ResidualBlock(256)
        self.res3=ResidualBlock(256)
        self.conv5 = CINR(256,256)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)  # stride=2的上采样
        self.conv6 = CINR(128,128)
        self.deconv1 = nn.ConvTranspose2d(256, 64, 4, 2, 1)  # stride=2的上采样
        self.conv7 = CINR(64,64)
        self.conv8 = CINR(128,64)
        self.PA=PALayer(64)
        self.CA=CALayer(64)
        self.conv9 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1)
        self.context=Contextattention(channels=16)
        self.dgm1=DGM(64,64)
        self.dgm2=DGM(128,128)
        self.dgm3=DGM(256,256)
        
        
    def forward(self, x):
        z = self.conv_in(x)#16*256*256
        context=self.context(z)
        z = self.conv1(z)#64*256*256 
        z1 = self.conv2(z)#64*256*256
        z = self.max1(z1)#64*128*128
        z2= self.conv3(z)#128*128*128
        z=self.max2(z2)#128*64*64
        z3=self.conv4(z)#256*64*64
        z=self.res1(z3)#256*64*64
        z=self.res2(z)#256*64*64
        z=self.res3(z)#256*64*64
        z4=self.conv5(z)#256*64*64
        z4=self.dgm3(z3,z4)
        z=torch.cat((z3,z4), dim=1)#512*64*64
        z=self.deconv2(z)#128*128*128
        z5=self.conv6(z)#128*128*128
        z5=self.dgm2(z2,z5)
        z=torch.cat((z2,z5), dim=1)#256*128*128
        z=self.deconv1(z)#64*256*256
        z6=self.conv7(z)#64*256*256
        z6=self.dgm1(z1,z6)
        z=torch.cat((z1,z6), dim=1)#128*256*256
        z=self.conv8(z)#64*256*256
        z=self.CA(z)
        z=self.PA(z)        
        z=F.relu(self.conv9(z))
        return z,context

class GCR(nn.Module):
    def __init__(self):
        super(GCR, self).__init__()
        self.conv_in = nn.Conv2d(16, 16, 1, 1, bias=True)
        self.conv_1  = CINR_PW(4,4)
        self.conv_2  = CINR_PW(4,4)
        self.conv_3  = CINR_PW(4,4)
        self.ad1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 1, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 4, 1, padding=0, bias=False),
            nn.Sigmoid())
        self.ad2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 1, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 4, 1, padding=0, bias=False),
            nn.Sigmoid())
        self.ad3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 1, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 4, 1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv_in(x)
        g1, g2, g3, g4= torch.chunk(x, 4, dim=1)
        a1 = self.ad1(g1)
        g1 = self.conv_1(g1)
        g1 = g1*a1
        a2 = self.ad2(g2)
        g2 = self.conv_2(g2)
        g2 = g2*a2
        a3 = self.ad3(g3)
        g3 = self.conv_3(g3)
        g3 = g3*a3
        g4 = g1 + g2 + g3 + g4
        out = torch.cat((g1,g2,g3,g4),dim=1)
        return out

class CBM(torch.nn.Module):
    def __init__(self):
        super(CBM, self).__init__()
        self.cinr1 = CINR_3D(1, 4)
        self.cinr2 = CINR_3D(4, 16)
        self.cinr4 = CINR_3D(3, 1)
        self.CINR1=nn.Sequential(
            CINR(3,16),
            CINR(16,16),
            CINR(16,3)
        )        
        self.ca = nn.Sequential(
            CINR(3,16),
            CAM_Module(16),
            nn.Sigmoid()
        )

    def forward(self, x):
        cbm=x
        x = torch.unsqueeze(x,1)
        x = self.cinr2(self.cinr1(x))
        x = torch.transpose(x, 1, 2)
        x = self.cinr4(x)
        out = torch.squeeze(x,1)
        out=out+self.ca(cbm)*out     
        return out

class CANet(nn.Module):
    def __init__(self, channelnums=16):
        super(CANet, self).__init__()
        self.cbm = CBM()
        self.drm = DRM()
        self.aff =  AFF(channelnums)
        self.gcr = GCR()
        self.conv = nn.Conv2d(channelnums, 3, 1, 1, bias=True)
        self.pa = PALayer(channelnums)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(channelnums, 3, 1, 1, bias=True)

    def forward(self, Input):
        colorfeatures= self.cbm(Input)
        detailfeatures,ca = self.drm(Input)
        fusion = self.aff(colorfeatures,detailfeatures,ca)
        out1 = self.act(fusion)
        out2 = self.conv(out1)
        out = self.gcr(out1)
        out = self.pa(out)
        out = self.act(out)
        out = self.out(out)
        return out, out2
        
class mynet(nn.Module):
    def __init__(self, opt):
        super(mynet, self).__init__()
        self.device = torch.device(opt.device)
        self.CANet = CANet().to(self.device)
        self.Lc = CharbonnierLoss().to(self.device)
        self.Lp = Perceptual_loss13().to(self.device)
        self.Lssim = SSIM().to(self.device)

    def forward(self, Input):
        self.Input = Input
        self.out1,self.out2 = self.CANet.forward(Input)
        return self.out1

    def elbo(self, target):
        self.lc = self.Lc(self.out1 , target)
        self.lp = self.Lp(self.out1 , target)
        self.lsps = self.Lp(self.out2 , target)
        self.lssim = self.Lssim(self.out1 , target)
        return self.lc + 0.2*self.lp + 0.2*self.lsps - 0.5*self.lssim
