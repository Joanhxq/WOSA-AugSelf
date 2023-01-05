import os
import argparse
import torch
import glob
import numpy as np

from unet import Unet
from utils import *


def seg_test(args, din_level, din_skip_level):

    model = Unet(args, din_level, din_skip_level)
    model = model.cuda()
    state_dict = torch.load(args.weight_path, map_location=lambda storage, loc:storage)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    if args.dataset == "fundus":
        style = load_image(args.style_path, args.image_size, gray=False)
    else:
        style = load_image(args.style_path, args.image_size)
    style = style.unsqueeze(0).cuda()
    
    dice_lst = []
    con_lst = []
    jac_lst = []
    hdb_lst = []
    asd_lst = []

    img_lst = glob.glob(f"{args.image_dir}/*")
    for img_path in img_lst:
        lab_path = os.path.join(args.label_dir, os.path.basename(img_path))
        if args.dataset == "fundus":
            image = load_image(img_path, args.image_size, gray=False)
        else:
            image = load_image(img_path, args.image_size)
        label = load_label(lab_path, args.image_size).squeeze(0).numpy()
        image = image.unsqueeze(0).cuda()
        logits = model(image, style)
        pred = np.argmax(logits.data.cpu().numpy(), 1)
        pred = np.squeeze(pred, 0).astype('uint8')
        pred = get_mask(pred)  ## postprocess
        
        pred = pred.astype('int64')
        dice = dc(pred, label) * 100
        con = d2c(dice)
        jaccard = mod_jc(pred, label) * 100
        Hdb = mod_hd95(pred, label)
        Asd = mod_asd(pred, label)

        dice_lst.append(dice)
        con_lst.append(con)
        jac_lst.append(jaccard)
        hdb_lst.append(Hdb)
        asd_lst.append(Asd)

    dice_mean = sum(dice_lst) / len(dice_lst)
    con_mean = sum(con_lst) / len(con_lst)
    jac_mean = sum(jac_lst) / len(jac_lst)
    hdb_mean = sum(hdb_lst) / len(hdb_lst)
    asd_mean = sum(asd_lst) / len(asd_lst)

    print(len(dice_lst))
    print(f'Dice: {dice_mean:.2f}')
    print(f'Con: {con_mean:.2f}')
    print(f'jaccard: {jac_mean:.2f}')
    print(f'hdb: {hdb_mean:.2f}')
    print(f'asd: {asd_mean:.2f}')


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    parser = argparse.ArgumentParser()
    ## =========================
    parser.add_argument("--dataset", default="fundus", help="Name of the dataset")
    parser.add_argument('--image_dir', type=str, default="dataset/fundus/IDRiD/image")
    parser.add_argument('--label_dir', type=str, default="dataset/fundus/IDRiD/label")
    parser.add_argument('--weight_path', type=str, default="save_output/fundus/weights/fundus_99.pth.tar")
    parser.add_argument('--style_path', type=str, default="drishtiGS_093.png")
    ## =========================
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--adain', action='store_true', default=False)
    parser.add_argument('--osa', action='store_true', default=False)
    parser.add_argument('--wosa', action='store_true', default=False)
    args = parser.parse_args()

    if args.adain or args.osa or args.wosa:
        din_level = [5]
        din_skip_level = [3, 4]
    else:
        din_level = []
        din_skip_level = []
    print(args)
    print(din_level, din_skip_level)

    seg_test(args, din_level, din_skip_level)
