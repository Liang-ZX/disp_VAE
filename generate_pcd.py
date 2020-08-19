import torch
import os
from torch.utils.data import DataLoader
from roi_dataset import KittiDataset
from resnet_encoder import DispVAEnet
from pcd_dataset import write_pcd_from_ndarray
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

    cfg = dict(device=device, batch_size=1, measure_cnt=2500, generate_cnt=2500, latent_num=256,
               roi_img_path="./datasets/roi_result/left/", save_rec_path="./datasets/img_generate/val/", is_val=True,
               resnet_model="./model_ckpt/model_resnet2.pth", vae_model="./model_ckpt/model_final2.pth", split="val",
               pcd_info_path="./datasets/pcd_info/")

    if not os.path.isdir(cfg['save_rec_path']):
        os.mkdir(cfg['save_rec_path'])

    roi_dst = KittiDataset(cfg)
    cfg['dst_len'] = roi_dst.__len__()
    val_loader = DataLoader(dataset=roi_dst, batch_size=1, shuffle=True, num_workers=0)

    model = DispVAEnet(cfg)
    model.to(device)

    inference(model, cfg, val_loader)


def inference(model, cfg, val_loader):
    print("Start generating...")
    device = cfg["device"]
    pbar = tqdm(total=cfg['dst_len'])
    pbar.set_description("Generating Point Cloud")
    model.eval()
    with torch.no_grad():
        for i, (img_list, img_meta, mean_list, max_list) in enumerate(val_loader):
            img_id = img_meta["img_id"][0]
            tmp_path = cfg['save_rec_path'] + img_id + ".xyz"
            points = None
            for j, img in enumerate(img_list):
                img = img.to(device=device, dtype=torch.float)
                z_decoded = model(img)
                z_decoded = z_decoded[0].cpu().numpy()
                z_decoded = z_decoded * 2.2 + mean_list[j][0].numpy()  # max_list[j].numpy()
                if points is None:
                    points = z_decoded
                else:
                    points = np.concatenate((points, z_decoded), axis=0)
            write_pcd_from_ndarray(points, tmp_path)
            pbar.update(1)
            if i == 3:
                break
        pbar.close()
    print("Finish Generating Point Cloud")


if __name__ == "__main__":
    main()
