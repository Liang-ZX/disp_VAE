import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from STNnet import PointNetfeat, T_Net
from CapsuleNet import PrimaryPointCapsLayer
# from loss_function import compute_chamfer_loss

import warnings
warnings.filterwarnings('ignore')


class VAEencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_pts = cfg["measure_cnt"]
        latent_num = cfg["latent_num"]
        self.fc1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64, momentum=0.99, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)

        self.fc2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.99, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)

        self.ppc = PrimaryPointCapsLayer(prim_vec_size=16, num_points=n_pts)  # output: B * 1024 * 16

        self.mean_fc = nn.Conv1d(1024, 128, kernel_size=1)
        self.mean_bn = nn.BatchNorm1d(128, momentum=0.99, eps=0.001)
        self.mean_relu = nn.LeakyReLU(negative_slope=0.3)

        self.logvar_fc = nn.Conv1d(1024, 128, kernel_size=1)
        self.logvar_bn = nn.BatchNorm1d(128, momentum=0.99, eps=0.001)
        self.logvar_relu = nn.LeakyReLU(negative_slope=0.3)

        self.encoder_fc1 = nn.Linear(128*16, latent_num)
        self.encoder_bn1 = nn.BatchNorm1d(latent_num)
        self.encoder_fc2 = nn.Linear(128*16, latent_num)
        self.encoder_bn2 = nn.BatchNorm1d(latent_num)

        self.z_mean = torch.zeros(cfg['batch_size'], latent_num)
        self.z_log_var = torch.ones(cfg['batch_size'], latent_num)

    def forward(self, input_coordinates):
        batch_size = input_coordinates.size()[0]
        # input B * 3 * N
        x = self.relu1(self.bn1(self.fc1(input_coordinates)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.ppc(x)  # B * 1024 * 16

        z_mean = self.mean_relu(self.mean_bn(self.mean_fc(x))).view(batch_size, -1)  # B * (128 * 16)
        z_log_var = self.logvar_relu(self.logvar_bn(self.logvar_fc(x))).view(batch_size, -1)
        z_mean = self.encoder_bn1(self.encoder_fc1(z_mean))
        z_log_var = self.encoder_bn2(self.encoder_fc2(z_log_var))
        self.z_mean, self.z_log_var = z_mean, z_log_var

        return z_mean, z_log_var  # B * latent_num


class VAEdecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        latent_num = cfg["latent_num"]
        self.conv1 = nn.Conv1d(latent_num, latent_num, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(latent_num)
        self.conv2 = nn.Conv1d(latent_num, latent_num//2, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(latent_num//2)
        self.conv3 = nn.Conv1d(latent_num//2, latent_num//4, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(latent_num//4)

        self.decoder_fc = nn.Conv1d(latent_num//4, 3, kernel_size=1)

    def forward(self, latent_code):
        # input B * latent_num * N
        x = F.relu(self.bn1(self.conv1(latent_code)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.decoder_fc(x)
        points = x

        output = torch.tanh(points)  # B * 3 * N
        # output = torch.sigmoid(x)
        return output


class VAEnn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.stn = T_Net(k=3)
        self.encoder = VAEencoder(cfg)
        self.decoder = VAEdecoder(cfg)
        self.criterion = nn.MSELoss(reduction='none')
        # self.feat = PointNetfeat(global_feat=False, feature_transform=False)  # pointNet  B * 1088 * N
        self.apply(init_weights)

    def forward(self, input_coordinates):
        # input B * N * 3
        input_coordinates = input_coordinates.transpose(2, 1).contiguous()  # B * 3 * N
        # trans = self.stn(input_coordinates)
        # points = torch.bmm(input_coordinates.transpose(2, 1), trans)
        # points = points.transpose(2, 1)
        points = input_coordinates

        # pts_feature, trans, _, pts_transformed = self.feat(input_coordinates)
        z_mean, z_log_var = self.encoder(points)
        epsilon = torch.randn(z_mean.size()[0], z_mean.size()[1], self.cfg["generate_cnt"])
        # epsilon = torch.zeros_like(z_mean)
        if z_mean.is_cuda:
            epsilon = epsilon.cuda()
        latent_code = z_mean.unsqueeze(2) + torch.exp(z_log_var.unsqueeze(2)) * epsilon  # B * latent_num * N

        z_decoded = self.decoder(latent_code)  # B * 3 * N
        # z_decoded = z_decoded.transpose(2, 1)
        # z_decoded = torch.bmm(z_decoded, torch.inverse(trans))
        # z_decoded = z_decoded.transpose(2, 1).contiguous()
        loss = self.vae_loss(input_coordinates, z_decoded)

        if self.training:
            return z_decoded.transpose(2, 1).contiguous(), latent_code, loss
        else:
            return z_decoded.transpose(2, 1).contiguous(), z_mean, z_log_var

    def vae_loss(self, input_coordinates, z_decoded):
        batch_size = input_coordinates.size()[0]
        # input B * 3 * N
        x = input_coordinates.transpose(2, 1).contiguous()
        z = z_decoded.transpose(2, 1).contiguous()
        # reconstr_loss = compute_chamfer_loss(x, z)  # B * 1
        z = z.view(batch_size, -1)
        x = x.view(batch_size, -1)
        reconstr_loss = torch.mean(self.criterion(z, x), dim=-1)
        # KL-loss
        z_log_var = self.encoder.z_log_var  # B * latent_num
        z_mean = self.encoder.z_mean
        latent_loss = -5e-4 * torch.mean(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=-1)  # KLDivloss
        loss = torch.mean(reconstr_loss + latent_loss)

        return loss


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)  # kaiming高斯初始化
        nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        pass
    elif classname.find('Conv') != -1:
        nn.init.constant_(m.bias, 0)