import torch
import os
from MeshDataset import PcdDataset, write_ndarray
from VAEnet import VAEnn
from torch.utils.data import DataLoader
from LatentDataset import write_latent_code
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

    cfg = dict(device=device, batch_size=20, measure_cnt=2500, generate_cnt=2500, latent_num=128 * 3,
               data_locate="./datasets/pcd_result", save_latent_path="./datasets/latent_result",
               model_path="./model_ckpt/model_final3.pth", is_val=True)

    if not os.path.isdir(cfg['save_latent_path']):
        os.mkdir(cfg['save_latent_path'])

    pcd_dst = PcdDataset(cfg)
    dst_len = pcd_dst.__len__()
    val_loader = DataLoader(dataset=pcd_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)

    model = VAEnn(cfg)
    model.to(device)

    # load from model_final
    # model.load_state_dict(torch.load(cfg['model_path']))  # cpu train
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['model_path']).items()})
    model.eval()

    inference(model, cfg, val_loader, dst_len)


def inference(model, cfg, val_loader, dst_len):
    device = cfg['device']
    print("Start generating...")
    pbar = tqdm(total=dst_len)
    pbar.set_description("Generating Latent Code")
    model.eval()
    with torch.no_grad():
        for i, (pcd_batch, mean_ref, max_ref, pcd_meta) in enumerate(val_loader):
            # for j in range(mean_ref.shape[0]):
            #     file_path = "./datasets/pcd_info/" + pcd_meta['img_id'][j]
            #     file_name = "/" + pcd_meta['car_id'][j] + ".txt"
            #     if not os.path.isdir(file_path):
            #         os.mkdir(file_path)
            #     array = torch.cat((mean_ref[j], max_ref[j]), dim=0).numpy()
            #     write_ndarray(array, file_path + file_name)
            #     pbar.update(1)

            pcd_batch = pcd_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            z_decoded, z_mean, z_log_var = model(pcd_batch)
            # z_decoded = z_decoded * (max_batch-mean_batch) + mean_batch
            # z_decoded = z_decoded.cpu().numpy()
            latent = torch.cat((z_mean.unsqueeze(2), z_log_var.unsqueeze(2)), dim=2)  # 第一列是mean, 第二列是log_var
            latent = latent.cpu().numpy()
            for j in range(z_mean.shape[0]):
                file_path = cfg['save_latent_path'] + "/" + pcd_meta['img_id'][j]
                file_name = "/" + pcd_meta['car_id'][j] + ".txt"
                if not os.path.isdir(file_path):
                    os.mkdir(file_path)
                write_latent_code(latent[j], file_path + file_name)
                pbar.update(1)
        pbar.close()
    print("Finish Generating Latent Code")


if __name__ == "__main__":
    main()
