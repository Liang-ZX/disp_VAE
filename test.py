import torch
import os
from torch.utils.data import DataLoader
from roi_dataset import KittiRoiDataset
from resnet_encoder import DispVAEnet
from mesh_dataset import write_pcd_from_ndarray
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = dict(device=device, batch_size=16, measure_cnt=2500, generate_cnt=2500, latent_num=256,
               roi_img_path="./datasets/roi_result/left/", save_pcd_path="./datasets/pcd_generate/val/",
               resnet_model="./model_ckpt/model_resnet1.pth", vae_model="./model_ckpt/model_final2.pth", is_val=True)

    if not os.path.isdir(cfg['save_pcd_path']):
        os.mkdir(cfg['save_pcd_path'])

    roi_dst = KittiRoiDataset(cfg)
    cfg['dst_len'] = roi_dst.__len__()
    val_loader = DataLoader(dataset=roi_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)

    model = DispVAEnet(cfg)
    model.to(device)

    inference(model, cfg, val_loader)


def inference(model, cfg, val_loader):
    print("Start generating...")
    pbar = tqdm(total=cfg['dst_len'])
    pbar.set_description("Generating Point Cloud")
    model.eval()
    with torch.no_grad():
        for i, (img, _, _, img_meta) in enumerate(val_loader):
            z_decoded = model(img)
            z_decoded = z_decoded.cpu().numpy()
            for j in range(z_decoded.shape[0]):
                img_id = img_meta['img_id'][j]
                car_id = img_meta['car_id'][j]
                tmp_path = cfg['save_pcd_path'] + img_id
                if not os.path.isdir(tmp_path):
                    os.mkdir(tmp_path)
                write_pcd_from_ndarray(z_decoded[j], tmp_path + "/" + car_id + ".xyz")
            pbar.update(cfg['batch_size'])
            if i > 1:
                break
        pbar.close()
    print("Finish Generating Point Cloud")


if __name__ == "__main__":
    main()
