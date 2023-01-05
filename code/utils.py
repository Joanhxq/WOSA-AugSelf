import cv2
import torch
import math
import torch.nn as nn
import numpy as np
from torchvision import transforms
from medpy.metric.binary import dc, hd95, jc, asd
import torch.nn.functional as F


def load_image(image_path, image_size, gray=True):
    if gray:
        image = cv2.imread(image_path, 0)
    else:
        image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    image = np.float32(image)
    image_mean = np.mean(image, axis=(0, 1), keepdims=True)
    image_std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - image_mean) / image_std
    image = transforms.ToTensor()(image)
    if gray:
        image = torch.cat((image, image, image), dim=0)
    return image


def load_label(label_path, image_size):
    label = cv2.imread(label_path, 0)
    label = cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    label = label // 255
    label = np.float32(label)
    label = transforms.ToTensor()(label)
    return label


def calc_mean_std(feat, eps=1e-5):
    n, c, h, w = feat.size()
    feat_view = feat.view(n, c, -1)
    feat_mean = torch.mean(feat_view, dim=2).view(n, c, 1, 1)
    feat_std = (torch.std(feat_view, dim=2) + eps).view(n, c, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    content_feat_norm = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return style_std.expand(size) * content_feat_norm + style_mean.expand(size)


def osa(cf, sf):
    device = cf.device
    b, c, h, w = cf.size()
    cf_view = cf.view(b, c, -1)
    sf_view = sf.view(b, c, -1)
    idx = torch.arange(0, c*h*w, step=h*w, device=device).reshape(-1, 1).unsqueeze(0)
    cf_ind = torch.sort(torch.sort(cf_view)[1])[1]
    sf_ind = torch.sort(sf_view)[1]
    cf_ind, sf_ind = cf_ind + idx, sf_ind + idx
    cf_ind_fatten, sf_ind_fatten = cf_ind.reshape(-1), sf_ind.reshape(-1)
    csf_ind = sf_ind_fatten[cf_ind_fatten]
    csf = sf_view.reshape(-1)[csf_ind].reshape(b, c, h, w)
    return csf


def wosa(cf, sf, kh, kw, sh, sw):
    device = cf.device
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64
    cf_unfold = cf.unfold(2, kh, sh).unfold(3, kw, sw)
    cf_pes = cf_unfold.reshape(b, c, -1, kh, kw)
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)  # .unfold(dim, size, step)
    sf_pes = sf_unfold.reshape(b, c, -1, kh, kw)
    p_num = cf_pes.shape[2]
    cf_pes = cf_pes.reshape(b, -1, kh*kw)
    sf_pes = sf_pes.reshape(b, -1, kh*kw)
    idx = torch.arange(0, c*p_num*kh*kw, step=kh*kw, device=device).reshape(-1, 1).unsqueeze(0)
    cf_ind = torch.sort(torch.sort(cf_pes)[1])[1]
    sf_ind = torch.sort(sf_pes)[1]
    cf_ind, sf_ind = cf_ind + idx, sf_ind + idx
    cf_ind, sf_ind = cf_ind.reshape(-1), sf_ind.reshape(-1)
    csf_ind = sf_ind[cf_ind]
    csf = sf_pes.reshape(-1)[csf_ind].reshape(b, c*p_num, kh*kw)
    csf = csf.reshape(b, c, -1, kh*kw)
    csf = csf.permute(0, 1, 3, 2)
    csf = csf.reshape(b, -1, p_num)
    overlap = F.fold(torch.ones_like(csf), (h, w), (kh, kw), stride=(sh, sw))
    csf = F.fold(csf, (h, w), (kh, kw), stride=(sh, sw))
    csf = csf / overlap
    return csf


def wosa_batch(cf, sf, kh, kw, sh, sw):
    device = cf.device
    b, c, h, w = cf.size()
    cf_unfold = cf.unfold(2, kh, sh).unfold(3, kw, sw)
    cf_pes = cf_unfold.reshape(b, c, -1, kh, kw)
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
    sf_pes = sf_unfold.reshape(b, c, -1, kh, kw)
    p_num = cf_pes.shape[2]
    cf_pes = cf_pes.reshape(b, -1, kh*kw)
    sf_pes = sf_pes.reshape(b, -1, kh*kw)
    idx_1 = torch.arange(0, b*c*p_num*kh*kw, step=c*p_num*kh*kw, device=device).reshape(-1, 1, 1)
    idx_2 = torch.arange(0, c*p_num*kh*kw, step=kh*kw, device=device).repeat(b, 1).unsqueeze(-1)
    cf_ind = torch.sort(torch.sort(cf_pes)[1])[1]
    sf_ind = torch.sort(sf_pes)[1]
    cf_ind, sf_ind = cf_ind + idx_1 + idx_2, sf_ind + idx_1 + idx_2
    cf_ind_flatten, sf_ind_flatten = cf_ind.reshape(-1), sf_ind.reshape(-1)
    csf_ind = sf_ind_flatten[cf_ind_flatten]
    csf = sf_pes.reshape(-1)[csf_ind].reshape(b, c*p_num, kh*kw)
    csf = csf.reshape(b, c, -1, kh*kw)
    csf = csf.permute(0, 1, 3, 2)
    csf = csf.reshape(b, -1, p_num)
    overlap = F.fold(torch.ones_like(csf), (h, w), (kh, kw), stride=(sh, sw))
    csf = F.fold(csf, (h, w), (kh, kw), stride=(sh, sw))
    csf = csf / overlap
    return csf


def get_mask(mask):
    contours, _ = cv2.findContours(mask, 1, 2)
    num = len(contours)
    areas = []
    for i in range(0, num):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        areas.append(area)
    if areas:
        areas_tar = sorted(areas)[:-2]
        for idx in areas_tar:
            id = areas.index(idx)
            cntMax = contours[id]
            cv2.drawContours(mask, [cntMax], 0, 0, -1)
        return mask
    else:
        return mask


def mod_hd95(pred, label):
    _sum1, _sum2 = np.sum(pred), np.sum(label)
    if _sum1 > 0 and _sum2 > 0:
        Hd95 = hd95(pred, label)
    else:
        Hd95 = 0.0  # Correct (no annotation)
    return Hd95


def mod_asd(pred, label):
    _sum1, _sum2 = np.sum(pred), np.sum(label)
    if _sum1 > 0 and _sum2 > 0:
        Asd = asd(pred, label)
    else:
        Asd = 0.0  # Correct (no annotation)
    return Asd


def mod_jc(pred, label):
    try:
        Jac = jc(pred, label)
    except ZeroDivisionError:
        Jac = 0.0
    return Jac


def d2c(dice):
    if dice > 1:
        dice /= 100
        con = (3 * dice - 2) / dice
    else:
        con = 0
    return con * 100


def calc_metrics_training(pred, label, cfg):
    dice_lst, hdb_lst = {key:[] for key in cfg.class_names.values()}, {key:[] for key in cfg.class_names.values()}
    for i in range(pred.shape[0]):  ## batch
        for c in range(1, len(cfg.class_names)+1):
            pred_temp = np.copy(pred[i])
            pred_temp[pred_temp != c] = 0
            label_temp = np.copy(label[i])
            label_temp[label_temp != c] = 0
            dice_temp = dc(pred_temp, label_temp)
            hdb_temp = mod_hd95(pred_temp, label_temp)
            dice_lst[cfg.class_names[c]].append(dice_temp)
            hdb_lst[cfg.class_names[c]].append(hdb_temp)
    dice_lst = {key: np.mean(val) for key, val in dice_lst.items()}
    hdb_lst  = {key: np.mean(val) for key, val in hdb_lst.items()}
    return dice_lst, hdb_lst


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val=0, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def focal_loss(inputs, targets):
    gamma = 2.
    alpha = 0.25
    class_num = 4
    p1 = torch.where(targets == 1, inputs, torch.ones_like(inputs))
    p0 = torch.where(targets == 0, inputs, torch.zeros_like(inputs))
    p1 = torch.clamp(p1, 1e-6, .999999)
    p0 = torch.clamp(p0, 1e-6, .999999)
    return (-torch.sum((1 - alpha) * torch.log(p1)) - torch.sum(alpha * torch.pow(p0, gamma) * torch.log(torch.ones_like(p0) - p0))) / float(class_num)


def dice_loss(inputs, targets):
    smooth = 1.
    iflaten = inputs.reshape(-1)
    tflaten = targets.reshape(-1)
    intersection = (iflaten * tflaten).sum()
    return 1 - ((2. * intersection + smooth) / (iflaten.sum() + tflaten.sum() + smooth))


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        beta = 0.001
        return beta * focal_loss(inputs, targets) + dice_loss(inputs, targets)


def rotate_map(mask, scale=1, angle=0):  ## mask: [N, C, H, W]
    if angle != 0:
        angle = angle * math.pi / 180
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle),  0]
        ], dtype=torch.float).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), mask.size(), align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)

    if scale != 1:
        theta = torch.tensor([
            [1/scale, 0, 0],
            [0, 1/scale, 0]
        ], dtype=torch.float).cuda()
        grid = F.affine_grid(theta.unsqueeze(0), mask.size(), align_corners=True)
        mask = F.grid_sample(mask, grid, align_corners=True)
    return mask


def move_map(mask, w_s=0, h_s=0):  ## mask: [C, H, W]
    h, w = mask.shape[2:]
    theta = torch.tensor([
        [1, 0, -w_s*2/w],
        [0, 1, -h_s*2/h]
    ], dtype=torch.float).cuda()
    grid = F.affine_grid(theta.unsqueeze(0), mask.size(), align_corners=True)
    mask = F.grid_sample(mask, grid, align_corners=True)
    return mask


def img_norm(image):
    image = np.float32(image)
    image_mean = np.mean(image, axis=(0, 1), keepdims=True)
    image_std = np.std(image, axis=(0, 1), keepdims=True)
    image = (image - image_mean) / image_std
    return image


def load_image_fundus(image_path, image_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img = img_norm(img)
    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
    img_1, img_2, img_3, img_4 = rotate_map(img, 1, angle=5), rotate_map(img, 1, angle=-5), torch.flip(img, [2]), torch.flip(img, [3])
    img_all = torch.cat((img, img_1, img_2, img_3, img_4), dim=0)
    return img_all


def mask_inv_trans_fundus(mask):
    mask_0, mask_1, mask_2, mask_3, mask_4 = mask[0], mask[1], mask[2], mask[3], mask[4]
    mask_1, mask_2, mask_3, mask_4 = rotate_map(mask_1.unsqueeze(0), 1, angle=-5).squeeze(0), \
                                     rotate_map(mask_2.unsqueeze(0), 1, angle=5).squeeze(0), \
                                     torch.flip(mask_3.unsqueeze(0), [2]).squeeze(0), \
                                     torch.flip(mask_4.unsqueeze(0), [3]).squeeze(0)
    mask_new = torch.stack([mask_0, mask_1, mask_2, mask_3, mask_4], dim=0)
    return mask_new


def mask_trans_fundus(mask):
    mask_0, mask_1, mask_2, mask_3, mask_4 = mask[0], mask[1], mask[2], mask[3], mask[4]
    mask_1, mask_2, mask_3, mask_4 = rotate_map(mask_1.unsqueeze(0), 1, angle=5).squeeze(0), \
                                     rotate_map(mask_2.unsqueeze(0), 1, angle=-5).squeeze(0), \
                                     torch.flip(mask_3.unsqueeze(0), [2]).squeeze(0), \
                                     torch.flip(mask_4.unsqueeze(0), [3]).squeeze(0)
    # mask[1], mask[2], mask[3], mask[4] = mask_1, mask_2, mask_3, mask_4
    mask_new = torch.stack([mask_0, mask_1, mask_2, mask_3, mask_4], dim=0)
    return mask_new


def gen_GUMs(mask):
    mask_sum = torch.sum(mask, dim=0)  ## (1, 400, 400)
    gums_all = torch.zeros((mask_sum.shape[0], 4, mask_sum.shape[1], mask_sum.shape[2])).cuda()  ## (1, 400, 400)
    gums_zero = torch.zeros_like(mask_sum[0]).cuda()
    gums_one = torch.ones_like(mask_sum[0]).cuda()
    for i in range(mask_sum.shape[0]):
        mask_sum_temp = mask_sum[i]
        gum_1 = torch.where(mask_sum_temp<0, gums_zero, mask_sum_temp)
        gum_1 = torch.where(gum_1>1, gums_one, gum_1)
        gum_2 = torch.where((mask_sum_temp-1)<0, gums_zero, (mask_sum_temp-1))
        gum_2 = torch.where(gum_2>1, gums_one, gum_2)
        gum_3 = torch.where((mask_sum_temp-2)<0, gums_zero, (mask_sum_temp-2))
        gum_3 = torch.where(gum_3>1, gums_one, gum_3)
        gum_5 = torch.where((mask_sum_temp-4)<0, gums_zero, (mask_sum_temp-4))
        gum_5 = torch.where(gum_5>1, gums_one, gum_5)
        gums_all[i][0], gums_all[i][1], gums_all[i][2], gums_all[i][3] = gum_1, gum_2, gum_3, gum_5
    return gums_all


def my_EMD(u_values, v_values):
    # p = 1
    # u_values = torch.from_numpy(u_values).float().cuda()
    # v_values = torch.from_numpy(v_values).float().cuda()

    u_values = torch.sort(u_values.data).values
    v_values = torch.sort(v_values.data).values

    all_values = torch.cat([u_values, v_values])
    all_values = all_values.sort().values

    # Compute the differences between pairs of successive values of u and v.
    deltas = all_values[1:] - all_values[:-1]

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = torch.searchsorted(u_values, all_values[:-1], right=True)
    v_cdf_indices = torch.searchsorted(v_values, all_values[:-1], right=True)

    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.shape[0]
    v_cdf = v_cdf_indices / v_values.shape[0]

    return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas)).cpu().numpy()

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    # if p == 1:
    #     torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas))
    # if p == 2:
    #     torch.sqrt(torch.sum(torch.multiply(torch.square(u_cdf - v_cdf), deltas)))


def get_one_hot_batch(label, class_num=2):  # b x h x w
    device = label.device
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(class_num, device=device)
    ones = ones.index_select(0, label)
    size.append(class_num)
    label = ones.view(*size)
    return label.permute(0, 3, 1, 2)


