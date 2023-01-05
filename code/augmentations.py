import random
from albumentations import HorizontalFlip, VerticalFlip, ShiftScaleRotate


class Compose(object):
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, label):
        if (random.random() < self.p):
            for t in self.transforms:
                img_lab = t(image=image, mask=label)
                image, label = img_lab['image'], img_lab['mask']
        return image, label


def do_nothing(image=None, mask=None):
    img_lab = {}
    img_lab['image'], img_lab['mask'] = image, mask
    return img_lab


def enable_if(condition, obj):
    return obj if condition else do_nothing


class FundusAugmentation(object):
    """ Transform to be used during training. """
    def __init__(self, cfg, p=1.0):
        self.augment = Compose([
            enable_if(cfg.augment_random_verticalflip, VerticalFlip(p=0.5)),
            enable_if(cfg.augment_random_horizontalflip, HorizontalFlip(p=0.5)),
            enable_if(cfg.augment_random_rotate, ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, border_mode=0, p=0.5)),
        ], p=p)

    def __call__(self, image, label):
        image, label = self.augment(image, label)
        return image, label


class XrayAugmentation(object):
    """ Transform to be used during training. """
    def __init__(self, cfg, p=1.0):
        self.augment = Compose([
            enable_if(cfg.augment_random_horizontalflip, HorizontalFlip(p=0.5)),
            enable_if(cfg.augment_random_shiftscale, ShiftScaleRotate(rotate_limit=0.0, border_mode=0, p=0.5)),
        ], p=p)

    def __call__(self, image, label):
        image, label = self.augment(image, label)
        return image, label


class USAugmentation(object):
    """ Transform to be used during training. """
    def __init__(self, cfg, p=1.0):
        self.augment = Compose([
            enable_if(cfg.augment_random_horizontalflip, HorizontalFlip(p=0.5)),
            enable_if(cfg.augment_random_shiftscalerotate, ShiftScaleRotate(border_mode=0, p=0.5)),
        ], p=p)

    def __call__(self, image, label):
        image, label = self.augment(image, label)
        return image, label

