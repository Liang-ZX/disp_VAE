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


def read_image(file_path):
    img = Image.open(file_path)
    return img


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, roi_root_dir, latent_root_dir, transforms=None):
        super().__init__()
        self.roi_root_dir = roi_root_dir
        self.latent_root_dir = latent_root_dir
        self.latent_names = glob.glob(self.latent_root_dir + "/*/*.txt")

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
        return len(self.latent_names)

    def __getitem__(self, index):
        latent_path = self.latent_names[index]
        (filepath, tempfilename) = os.path.split(latent_path)
        img_id = os.path.split(filepath)[1]
        car_id = os.path.splitext(tempfilename)[0]
        img_path = self.roi_root_dir + "/"+img_id+"/"+car_id+".png"
        img_meta = dict(img_id=img_id, car_id=car_id)
        
        img = read_image(img_path)
        img = self.transform(img)

        z_mean, z_log_var = read_latent_code(latent_path)
        return img, z_mean, z_log_var, img_meta
