import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
from PIL import Image
import glob
import os


class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self.root_dir = "./datasets/roi_result/left"
        self.file_names = glob.glob(self.root_dir + "/*/*.png")

        if transforms is not None:
            self.transform = transforms
        else:
            self.transform = T.Compose([
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406],
                #             std=[0.229, 0.224, 0.225]) #TODO
                ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        (filepath, tempfilename) = os.path.split(img_path)
        img_meta = dict(img_id=os.path.split(filepath)[1], car_id=os.path.splitext(tempfilename)[0])

        img = read_image(img_path)
        img = self.transform(img)
        return img


def read_image(file_path):
    img = Image.open(file_path)
    return img
