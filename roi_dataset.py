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
    mean_pcd = array[:3]
    max_pcd = array[3]
    return torch.tensor(mean_pcd).view(1, 3), torch.tensor(max_pcd)


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
        z_mean, z_log_var = 0, 0
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


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None):
        super().__init__()
        self.roi_root_dir = cfg['roi_img_path'] + cfg["split"] + "/"
        self.pcd_info_dir = cfg["pcd_info_path"]
        split_path = "/mnt/data/liangzx/KITTI/object/splits/" + cfg["split"] + ".txt"
        with open(split_path) as f:
            self.ids = np.loadtxt(f, dtype='str')

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
        return self.ids.shape[0]

    def __getitem__(self, index):
        img_id = self.ids[index]
        # img_names = glob.glob(self.roi_root_dir + img_id + "/*.png")
        info_names = glob.glob(self.pcd_info_dir + img_id + "/*.txt")
        img_list, mean_list, max_list = [], [], []
        for tmp_path in info_names:
            (filepath, tempfilename) = os.path.split(tmp_path)
            car_id = os.path.splitext(tempfilename)[0]
            img_names = self.roi_root_dir + img_id + "/" + car_id + ".png"
            mean_ref, max_ref = read_pcd_info(tmp_path)
            img = read_image(img_names)
            img_list.append(self.transform(img))
            mean_list.append(mean_ref)
            max_list.append(max_ref)

        img_meta = dict(img_id=img_id)
        return img_list, img_meta, mean_list, max_list
