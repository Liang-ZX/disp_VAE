import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
from PIL import Image
import glob
import os


def write_latent_code(array, save_path):
    with open(save_path, 'w') as f:
        np.savetxt(f, array, delimiter=' ', fmt='%f %f')


def read_latent_code(file_path):
    latent = np.loadtxt(file_path)
    z_mean = latent[:, 0]
    z_log_var = latent[:, 1]
    return torch.tensor(z_mean), torch.tensor(z_log_var)


def read_pcd_info(file_path):
    array = np.loadtxt(file_path)
    mean_pcd = array[0]
    max_pcd = array[1]
    return torch.tensor(mean_pcd).view(1, 3), torch.tensor(max_pcd).view(1, 3)


def read_image(file_path):
    img = Image.open(file_path)
    return img


class KittiRoiDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None):
        super().__init__()
        self.is_val = cfg["is_val"]
        self.roi_root_dir = cfg['roi_img_path']
        if not self.is_val:
            self.latent_root_dir = cfg['save_latent_path']
            self.latent_names = glob.glob(self.latent_root_dir + "train/*/*.txt")
        else:
            self.img_names = glob.glob(self.roi_root_dir + "val/*/*.png")

        if transforms is not None:
            self.transform = transforms
        else:
            self.transform = T.Compose([
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                ])
        
    def __len__(self):
        if self.is_val:
            return len(self.img_names)
        else:
            return len(self.latent_names)

    def __getitem__(self, index):
        z_mean, z_log_var = None, None
        if not self.is_val:
            latent_path = self.latent_names[index]
            (filepath, tempfilename) = os.path.split(latent_path)
            img_id = os.path.split(filepath)[1]
            car_id = os.path.splitext(tempfilename)[0]
            img_path = self.roi_root_dir + "train/" + img_id + "/" + car_id + ".png"
            z_mean, z_log_var = read_latent_code(latent_path)
        else:
            img_path = self.img_names[index]
            (filepath, tempfilename) = os.path.split(img_path)
            img_id = os.path.split(filepath)[1]
            car_id = os.path.splitext(tempfilename)[0]

        img_meta = dict(img_id=img_id, car_id=car_id)
        img = read_image(img_path)
        img = self.transform(img)

        return img, z_mean, z_log_var, img_meta


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.latent_root_dir = cfg['save_latent_path']
        self.latent_names = glob.glob(self.latent_root_dir + "/*/*.txt")
        self.pcd_info_dir = "./datasets/pcd_info"

    def __len__(self):
        return len(self.latent_names)

    def __getitem__(self, index):
        latent_path = self.latent_names[index]
        (filepath, tempfilename) = os.path.split(latent_path)
        img_id = os.path.split(filepath)[1]
        car_id = os.path.splitext(tempfilename)[0]
        img_meta = dict(img_id=img_id, car_id=car_id)

        mean_pcd, max_pcd = read_pcd_info(self.pcd_info_dir+"/"+img_id+"/"+car_id+".txt")
        z_mean, z_log_var = read_latent_code(latent_path)
        return z_mean, z_log_var, mean_pcd, max_pcd, img_meta
