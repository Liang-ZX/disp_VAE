import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
from pcd_dataset import PcdDataset, write_pcd_from_ndarray
from VAEnet import VAEnn
import warnings
warnings.filterwarnings('ignore')


def main():
    USE_GPU = True
    RUN_PARALLEL = True
    device_ids = [0, 1, 2, 3]
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.device_count() <= 1:
            RUN_PARALLEL = False
    else:
        device = torch.device('cpu')
        RUN_PARALLEL = False

    learning_rate = 1e-3
    learning_rate_decay = 0.3
    cfg = dict(device=device, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, epochs=4,
               print_every=20, save_every=2, batch_size=16, measure_cnt=2500, generate_cnt=2500, latent_num=256,
               data_locate="./datasets/pcd_result/", save_path="./model_ckpt/", save_pcd_path="./decode_result/",
               tensorboard_path="runs/train_visualization", is_val=False)

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])
    if not os.path.isdir(cfg['save_pcd_path']):
        os.mkdir(cfg['save_pcd_path'])
    writer = SummaryWriter(cfg['tensorboard_path'])

    pcd_dst = PcdDataset(cfg)
    train_loader = DataLoader(dataset=pcd_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = VAEnn(cfg)
    model.to(device)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("./model_ckpt/model_final.pth").items()})
    if RUN_PARALLEL:
        model = nn.DataParallel(model, device_ids=device_ids)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=learning_rate_decay)

    print("Start Training...")
    do_train(model, cfg, train_loader, optimizer, scheduler=None, writer=writer)


def do_train(model, cfg, train_loader, optimizer, scheduler, writer):
    print_every = cfg['print_every']
    device = cfg['device']
    model.train()
    for e in range(cfg['epochs']):
        for i, (pcd_batch, _) in enumerate(train_loader):
            pcd_batch = pcd_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            # normalize [-1,1]
            z_decoded, latent, loss = model(pcd_batch)

            if loss.is_cuda:
                loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                print('Epoch %d/5: Iteration %d, loss = %.4f' % (e+1, i, loss.item()))
                writer.add_scalar('training_loss', loss.item(), e)

        # scheduler.step()
        # if (e+1) % cfg['save_every'] == 0:
        #     file_path = cfg['save_path'] + "model_epoch" + str(e+1) + ".pth"
        #     torch.save({
        #         'epoch': e,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         # 'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': loss
        #     }, file_path)
        #     print("Save model "+file_path)
    torch.save(model.state_dict(), cfg['save_path']+"model_final.pth")
    print("Save final model "+cfg['save_path']+"model_final.pth")
    print("Finish Training")
    model.eval()
    with torch.no_grad():
        for i, (pcd_batch, mean_ref) in enumerate(train_loader):
            pcd_batch = pcd_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            z_decoded, _, _ = model(pcd_batch)
            z_decoded = z_decoded.cpu().numpy()
            for j in range(z_decoded.shape[0]):
                write_pcd_from_ndarray(z_decoded[j], cfg['save_pcd_path']+"test"+str(j)+".xyz")
            if i > 2:
                break
    print("Finish writing")


if __name__ == "__main__":
    main()
