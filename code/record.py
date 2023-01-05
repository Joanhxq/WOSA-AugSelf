from utils import *


def record_dice(mask_1, mask_2, mask_3, label, args, dice_mask):
    dice_1 = dc(mask_1, label) * 100
    dice_2 = dc(mask_2, label) * 100
    dice_3 = dc(mask_3, label) * 100
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice1.txt', 'a') as f:
        f.write(f'{dice_1:.6f} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice2.txt', 'a') as f:
        f.write(f'{dice_2:.6f} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice3.txt', 'a') as f:
        f.write(f'{dice_3:.6f} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice.txt', 'a') as f:
        f.write(f'{dice_mask:.6f} ')

def record_dice_fname(fname, args):
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice1.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice2.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice3.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice.txt', 'a') as f:
        f.write(f'{fname} ')

def record_dice_n(args):
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice1.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice2.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice3.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/dice/ss_{args.lr}_dice.txt', 'a') as f:
        f.write('\n')


def record_jac(mask_1, mask_2, mask_3, label, args):
    jac_1 = mod_jc(mask_1, label) * 100
    jac_2 = mod_jc(mask_2, label) * 100
    jac_3 = mod_jc(mask_3, label) * 100
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac1.txt', 'a') as f:
        f.write(f'{jac_1:.6f} ')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac2.txt', 'a') as f:
        f.write(f'{jac_2:.6f} ')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac3.txt', 'a') as f:
        f.write(f'{jac_3:.6f} ')

def record_jac_fname(fname, args):
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac1.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac2.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac3.txt', 'a') as f:
        f.write(f'{fname} ')

def record_jac_n(args):
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac1.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac2.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/jac/ss_{args.lr}_jac3.txt', 'a') as f:
        f.write('\n')


def record_hdb(mask_1, mask_2, mask_3, label, args):
    hdb_1 = mod_hd95(mask_1, label)
    hdb_2 = mod_hd95(mask_2, label)
    hdb_3 = mod_hd95(mask_3, label)
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb1.txt', 'a') as f:
        f.write(f'{hdb_1:.6f} ')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb2.txt', 'a') as f:
        f.write(f'{hdb_2:.6f} ')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb3.txt', 'a') as f:
        f.write(f'{hdb_3:.6f} ')

def record_hdb_fname(fname, args):
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb1.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb2.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb3.txt', 'a') as f:
        f.write(f'{fname} ')

def record_hdb_n(args):
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb1.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb2.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/hdb95/ss_{args.lr}_hdb3.txt', 'a') as f:
        f.write('\n')


def record_asd(mask_1, mask_2, mask_3, label, args):
    asd_1 = mod_asd(mask_1, label)
    asd_2 = mod_asd(mask_2, label)
    asd_3 = mod_asd(mask_3, label)
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd1.txt', 'a') as f:
        f.write(f'{asd_1:.6f} ')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd2.txt', 'a') as f:
        f.write(f'{asd_2:.6f} ')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd3.txt', 'a') as f:
        f.write(f'{asd_3:.6f} ')

def record_asd_fname(fname, args):
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd1.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd2.txt', 'a') as f:
        f.write(f'{fname} ')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd3.txt', 'a') as f:
        f.write(f'{fname} ')

def record_asd_n(args):
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd1.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd2.txt', 'a') as f:
        f.write('\n')
    with open(f'{args.save_dir}/asd/ss_{args.lr}_asd3.txt', 'a') as f:
        f.write('\n')
