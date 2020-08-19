import torch
import torchvision
import torch.nn as nn
from VAEnet import VAEnn


class ResnetEncoder(nn.Module):
    def __init__(self, c_dim):
        super().__init__()
        self.mean_encoder = Resnet18(c_dim)
        self.log_var_encoder = Resnet18(c_dim)

    def forward(self, x):
        z_mean = self.mean_encoder(x)
        z_log_var = self.log_var_encoder(x)
        return z_mean, z_log_var  # B * (256*2)


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


class DispVAEnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResnetEncoder(cfg["latent_num"] * 2)
        self.decoder = VAEnn(cfg, with_encoder=False)
        self.is_val = cfg["is_val"]
        if self.is_val:
            self.encoder.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(cfg['resnet_model']).items()})
            self.decoder.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['vae_model']).items()})
            # model.load_state_dict(torch.load(cfg['model_path']))  # cpu train

    def forward(self, img):
        z_mean, z_log_var = self.encoder(img)  # B * latent_num
        z_decoded, _, _ = self.decoder(z_mean, z_log_var)
        return z_decoded
