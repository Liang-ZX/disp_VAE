import torch
from torch.utils.data import DataLoader
from LatentDataset import KittiRoiDataset, write_latent_code
from resnetencoder import ResnetEncoder
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


def main():
    USE_GPU = True
    RUN_PARALLEL = True
    device_ids = [0, 1]
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.device_count() <= 1:
            RUN_PARALLEL = False
            pass
    else:
        device = torch.device('cpu')
        RUN_PARALLEL = False

    learning_rate = 1e-3
    learning_rate_decay = 0.3
    cfg = dict(device=device, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
               epochs=15, print_every=20, save_every=2, batch_size=50, roi_img_path="./datasets/roi_result/left",
               save_latent_path="./datasets/latent_result", save_path="./model_ckpt/", log_file="./log.txt",
               tensorboard_path="runs/train_visualization")

    train_dataset = KittiRoiDataset(cfg['roi_img_path'], cfg['save_latent_path'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = ResnetEncoder(128 * 3)
    model.to(device)
    if RUN_PARALLEL:
        model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9)

    print("Start Training...")
    do_train(model, cfg, train_loader, optimizer)


def do_train(model, cfg, train_loader, optimizer, scheduler=None):
    device = cfg['device']
    print_every = cfg['print_every']
    writer = SummaryWriter(cfg['tensorboard_path'])
    model.train()
    for e in range(cfg['epochs']):
        for i, data in enumerate(train_loader):
            img, gt_z_mean, gt_z_log_var, _ = data
            # epsilon = torch.randn(z_mean.size()[0])
            # latent_code = z_mean + torch.exp(z_log_var) * epsilon  # B * latent_num
            img = img.to(device=device, dtype=torch.float)
            gt_z_mean = gt_z_mean.to(device, dtype=torch.float)
            gt_z_log_var = gt_z_log_var.to(device, dtype=torch.float)
            pred_z_mean, pred_z_log_var = model(img)
            criterion = nn.MSELoss(reduction='mean')
            loss = criterion(pred_z_mean, gt_z_mean) + criterion(pred_z_log_var, gt_z_log_var)
            if loss.is_cuda:
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                writer.add_scalar('training_loss', loss.item(), e)
                print('Epoch %d/25: Iteration %d, loss = %.4f' % (e + 1, i, loss.item()))

    torch.save(model.state_dict(), cfg['save_path'] + "model_resnet.pth")
    print("Save final model " + cfg['save_path'] + "model_resnet.pth")
    print("Finish Training")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            img, _, _, img_meta = data
            img = img.to(device=device, dtype=torch.float)
            pred_z_mean, pred_z_log_var = model(img)
            latent = torch.cat((pred_z_mean.unsqueeze(2), pred_z_log_var.unsqueeze(2)), dim=2)  # 第一列是mean, 第二列是log_var
            latent = latent.cpu().numpy()
            for j in range(pred_z_mean.shape[0]):
                file_path = cfg['save_path'] + "result/" + img_meta['img_id'][j]
                file_name = "/" + img_meta['car_id'][j] + ".txt"
                if not os.path.isdir(file_path):
                    os.mkdir(file_path)
                write_latent_code(latent[j], file_path + file_name)
            if i > 1:
                break

    print("Finish writing")


if __name__ == "__main__":
    main()
