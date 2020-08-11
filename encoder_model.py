import torch
import torchvision
import torch.nn as nn


class ResnetEncoder(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.mean_encoder = Resnet18(c_dim)
        self.log_var_encoder = Resnet18(c_dim)

    def forward(self, x):
        z_mean = self.mean_encoder(x)
        z_log_var = self.log_var_encoder(x)
        return z_mean, z_log_var # B * 384


class Resnet18(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, c_dim)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet34(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.features = torchvision.models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, c_dim)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.features = torchvision.models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2048, c_dim)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.features = torchvision.models.resnet101(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2048, c_dim)

    def forward(self, x):
        net = self.features(x)
        out = self.fc(net)
        return out