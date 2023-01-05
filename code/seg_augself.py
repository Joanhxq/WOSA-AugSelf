import argparse
import os
import torch
import tqdm
import time
import datetime

from unet import Unet
from utils import *
from record import *


def main(args, din_level, din_skip_level):

    model = Unet(args, din_level, din_skip_level).cuda()
    model_dict = torch.load(args.weight_path, map_location=lambda storage, loc:storage)["state_dict"]
    
    style = load_image_fundus(args.style_path, args.image_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    criterion = DiceFocalLoss()

    fnames = os.listdir(args.image_dir)
    start = time.time()
    for fname in tqdm.tqdm(fnames):
        # fname = "drishtiGS_015.png"

        record_dice_fname(fname, args)
        record_jac_fname(fname, args)
        record_hdb_fname(fname, args)
        record_asd_fname(fname, args)

        model.load_state_dict(model_dict)
        model.eval()
        image_path = os.path.join(args.image_dir, fname)
        label_path = os.path.join(args.label_dir, fname)
        image = load_image_fundus(image_path, args.image_size)  ## b x c x h x w
        label = load_label(label_path, args.image_size).squeeze(0).numpy()
        dice_mask_pre, dice_mask = 1000, 0.0001
        dice_sub = dice_mask - dice_mask_pre
        cnt = 0
        dice_mask_all, dice_lab_all = [], []
        while ((cnt <= args.max_iter) and (dice_sub <= 0)):
            cnt += 1
            logits = model(image, style)
            mask = torch.softmax(logits, dim=1)  ## [5, 4, 400, 400]
            mask_target = mask[:, [1], ...]
            ## ======== inverse-transformation ========
            mask_target = mask_inv_trans(mask_target)

            ## ======== generate GUMs ========
            gums_all = gen_GUMs(mask_target)

            ## ======== proxy label ========
            proxy_idx = []
            for i in range(mask_target.shape[0]):  ## trans 1~5   mask_target: (5, 1, 400, 400)
                for j in range(mask_target.shape[1]):  ## channel 1
                    mask_temp = mask_target[i][j]
                    kl = np.array([my_EMD(mask_temp.reshape(-1), gums_all[j][0].reshape(-1)), my_EMD(mask_temp.reshape(-1), gums_all[j][1].reshape(-1)), my_EMD(mask_temp.reshape(-1), gums_all[j][2].reshape(-1))])
                    proxy_idx.append(np.argmin(kl))
                    mask_target[i][j] = gums_all[j][np.argmin(kl)]

            ## dice loss  gum_1 与 gum_5
            dice_mask = dice_loss(gums_all[0][0], gums_all[0][3])
            dice_mask_all.append(dice_mask)
            dice_sub = dice_mask - dice_mask_pre
            dice_mask_pre = dice_mask

            ## ======== proxy label ==== trans for SSL
            mask_target = mask_trans(mask_target)

            ## mask_target: (5, 1, 400, 400)
            mask_target = mask_target.data.cpu().numpy()
            for i in range(mask_target.shape[0]):  ## trans 1~5
                for j in range(mask_target.shape[1]):  ## channel
                    mask_temp = mask_target[i][j]
                    mask_temp = np.where(mask_temp>=args.noisy[proxy_idx[j*5+i]], 1, 0)
                    mask_temp = mask_temp.astype('uint8')
                    mask_temp = get_mask(mask_temp)
                    mask_target[i][j] = mask_temp

            fname = fname.split('.')[0]
            ## gums_all: (1, 4, 400, 400)
            gums_all = gums_all.data.cpu().numpy()
            for i in range(gums_all.shape[0]):  ## channel
                for j in range(gums_all.shape[1]-1):  ## gums
                    gum_tmp = gums_all[i][j]
                    gum_tmp = np.where(gum_tmp>=args.noisy[j], 1, 0)
                    gum_tmp = gum_tmp.astype('uint8')
                    gum_tmp = get_mask(gum_tmp)
                    cv2.imwrite(f"{args.save_dir}/vis/mask_{j+1}/{fname}_iter{cnt}.bmp", gum_tmp*255)
                    gums_all[i][j] = gum_tmp

            dice_lab = dc(gums_all[0][1], label)
            dice_lab_all.append(dice_lab)

            # mask_bg = 
            # mask[:, c_idx, ...] = mask_target
            mask = torch.from_numpy(mask_target).cuda().squeeze(1)
            mask = get_one_hot_batch(mask.long())

            loss = criterion(logits, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            record_dice(gums_all[0][0], gums_all[0][1], gums_all[0][2], label, args, dice_mask*100)
            record_jac(gums_all[0][0], gums_all[0][1], gums_all[0][2], label, args)
            record_hdb(gums_all[0][0], gums_all[0][1], gums_all[0][2], label, args)
            record_asd(gums_all[0][0], gums_all[0][1], gums_all[0][2], label, args)
        
        record_dice_n(args)
        record_jac_n(args)
        record_hdb_n(args)
        record_asd_n(args)

    return start


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    ## =========================
    parser.add_argument("--dataset", default="fundus", help="Name of the dataset")
    parser.add_argument('--image_dir', type=str, default="/data/hxq/SCI/CMPB/dataset/fundus/IDRiD/image")
    parser.add_argument('--label_dir', type=str, default="/data/hxq/SCI/CMPB/dataset/fundus/IDRiD/label")
    parser.add_argument('--weight_path', type=str, default="save_output/fundus/weights/fundus_99.pth.tar")
    parser.add_argument('--style_path', type=str, default="/data/hxq/SCI/CMPB/dataset/fundus/DrishtiGS/image/drishtiGS_093.png")
    parser.add_argument('--save_dir', type=str, default='./test_TT/D2I')
    ## =========================
    parser.add_argument('--noisy', type=list, default=[0.6, 0.6, 0.6])
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--wosa', action='store_true', default=False)
    parser.add_argument('--max_iter', type=int, default=10)
    args = parser.parse_args()
    args.adain, args.osa = False, False
    
    if args.wosa:
        din_level = [5]  # 1, 2, 3, 4, 5, 6, 7, 8, 9
        din_skip_level = [3, 4]  # 1, 2, 3, 4
    else:
        din_level = []
        din_skip_level = []
        args.save_dir = args.save_dir + "_ori"

    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(args)
    print(f"din_level: {din_level}, din_skip_level: {din_skip_level}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f'{args.save_dir}/dice', exist_ok=True)
    os.makedirs(f'{args.save_dir}/jac', exist_ok=True)
    os.makedirs(f'{args.save_dir}/hdb95', exist_ok=True)
    os.makedirs(f'{args.save_dir}/asd', exist_ok=True)
    for i in range(1, 4):  ## 1, 2, 3, 4, 5
        os.makedirs(f'{args.save_dir}/vis/mask_{i}', exist_ok=True)

    os.makedirs(f'{args.save_dir}/loss', exist_ok=True)

    start = main(args, din_level, din_skip_level)
    print(f'Used Time: {(time.time() - start) / 60}min {time.time() - start}s')
