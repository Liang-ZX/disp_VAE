import torch
import os
from VAEnet import VAEnn
from torch.utils.data import DataLoader
from LatentDataset import LatentDataset
from MeshDataset import write_pcd_from_ndarray
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

    cfg = dict(device=device, batch_size=10, measure_cnt=2500, generate_cnt=2500, latent_num=128 * 3,
               save_latent_path="./datasets/latent_result", save_pcd_path="./datasets/pcd_reconstruct",
               model_path="./model_ckpt/model_final3.pth")

    if not os.path.isdir(cfg['save_pcd_path']):
        os.mkdir(cfg['save_pcd_path'])

    latent_dst = LatentDataset(cfg)
    cfg['dst_len'] = latent_dst.__len__()
    val_loader = DataLoader(dataset=latent_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)

    model = VAEnn(cfg, with_encoder=False)
    model.to(device)

    # load from model_final
    # model.load_state_dict(torch.load(cfg['model_path']))  # cpu train
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['model_path']).items()})

    inference(model, cfg, val_loader)


def inference(model, cfg, val_loader):
    device = cfg['device']
    print("Start generating...")
    pbar = tqdm(total=cfg['dst_len'])
    pbar.set_description("Generating Point Cloud")
    model.eval()
    with torch.no_grad():
        for i, (z_mean, z_log_var, mean_pcd, max_pcd, img_meta) in enumerate(val_loader):
            z_mean = z_mean.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            z_log_var = z_log_var.to(device=device, dtype=torch.float)
            z_decoded, _, _ = model(z_mean, z_log_var)
            mean_pcd = mean_pcd.repeat(1, z_decoded.shape[1], 1).to(device=device, dtype=torch.float)
            max_pcd = max_pcd.repeat(1, z_decoded.shape[1], 1).to(device=device, dtype=torch.float)
            z_decoded = z_decoded * (max_pcd-mean_pcd) + mean_pcd
            z_decoded = z_decoded.cpu().numpy()
            for j in range(z_decoded.shape[0]):
                img_id = img_meta['img_id'][j]
                car_id = img_meta['car_id'][j]
                tmp_path = cfg['save_pcd_path'] + "/" + img_id
                if not os.path.isdir(tmp_path):
                    os.mkdir(tmp_path)
                write_pcd_from_ndarray(z_decoded[j], tmp_path + "/" + car_id + ".xyz")
            pbar.update(cfg['batch_size'])
        pbar.close()
    print("Finish Generating Point Cloud")


if __name__ == "__main__":
    main()