import argparse
import os
import logging
import sys
import torch
import glob
import tqdm
import datetime
from torch.utils.data import DataLoader

from config import *
from augmentations import *
from utils import *
from dataset import *
from unet import Unet


def main():

    ## ==== logging ====
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(args.save_dir, "logs", "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("cfg  = %s", cfg.to_dict())
    logging.info("args = %s", dict(args._get_kwargs()))

    model = Unet().cuda()
    model.init_weights()

    train_imgs = glob.glob(f"{cfg.train_dir}/*")
    if args.dataset == "fundus":
        dataset = FundusDataset(img_lst=train_imgs, cfg=cfg, transform=FundusAugmentation(cfg=cfg))
    elif args.dataset == "xray":
        dataset = GrayDataset(img_lst=train_imgs, cfg=cfg, transform=XrayAugmentation(cfg=cfg))
    elif args.dataset == "us":
        dataset = GrayDataset(img_lst=train_imgs, cfg=cfg, transform=USAugmentation(cfg=cfg))

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(cfg.max_epoch), eta_min=cfg.lr_min)
    criterion = DiceFocalLoss()

    loss_meter = AverageMeter()
    dice_meter = {i: AverageMeter() for i in dict_str}
    hd_meter  = {i: AverageMeter() for i in dict_str}
    logs_dict = {"LOSS": [], "DICE": {i: [] for i in dict_str}, "HD": {i: [] for i in dict_str}}

    print("Begin training!")

    for epoch in range(cfg.max_epoch):
        lr = scheduler.get_last_lr()[0]
        logging.info("Epoch: %d lr: %e", epoch, lr)
        model = model.train()
        for input, target in tqdm.tqdm(dataloader):
            input = input.cuda()
            target = target.cuda()
            logits = model(input)
            loss = criterion(logits, target)
            loss_meter.update(loss, cfg.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(torch.softmax(logits, 1), 1).data.cpu()
            label = torch.argmax(target, 1).data.cpu()
            dice_temp, hd_temp = calc_metrics_training(pred.numpy(), label.numpy(), cfg)
            for dict_idx in dict_str:
                dice_meter[dict_idx].update(dice_temp[dict_idx])
                hd_meter[dict_idx].update(hd_temp[dict_idx])
            
        logs_dict["LOSS"].append(loss_meter.avg.item())
        loss_meter.reset()
        for dict_idx in dict_str:
            logs_dict["DICE"][dict_idx].append(dice_meter[dict_idx].avg)
            logs_dict["HD"][dict_idx].append(hd_meter[dict_idx].avg)
            dice_meter[dict_idx].reset()
            hd_meter[dict_idx].reset()
        logging.info("Train--- Loss: %.8f  Dice: %s  HD: %s", logs_dict["LOSS"][-1], 
                        {key: round(val[-1], 8) for key, val in logs_dict["DICE"].items()}, 
                        {key: round(val[-1], 8) for key, val in logs_dict["HD"].items()})
        scheduler.step()
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "final_lr": scheduler.get_last_lr()[0]},
                    f"{args.save_dir}/weights/{args.dataset}_{epoch}.pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fundus", help="Name of the dataset")
    parser.add_argument("--save_dir", type=str, default="save_output")
    args = parser.parse_args()

    if args.dataset == "fundus":
        cfg = cfg_fundus.copy()
    elif args.dataset == "xray":
        cfg = cfg_xray.copy()
    elif args.dataset == "us":
        cfg = cfg_us.copy()
    args.save_dir = os.path.join(args.save_dir, args.dataset)
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(f"{args.save_dir}/weights", exist_ok=True)
    os.makedirs(f"{args.save_dir}/logs", exist_ok=True)

    dict_str = [i for i in cfg.class_names.values()]

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
