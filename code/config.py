
class Config(object):
    """ Config class for setting a config file. """
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)
    
    def copy(self, new_config_dict={}):
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)
        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, '=', v)

    def to_dict(self):
        config_dict = {}
        for k, v in vars(self).items():
            if isinstance(v, Config):
                for k_t, v_t in vars(v).items():
                    config_dict[f"{k}.{k_t}"] = v_t
            else:
                config_dict[k] = v
        return config_dict


## =========== configs of different datasets ===========
cfg_fundus = Config({
    "train_dir": "dataset/fundus/DrishtiGS/image",
    "label_dir": "dataset/fundus/DrishtiGS/label",
    "class_names": {1: "OD"},
    "class_maps": {255: 1},
    "image_size": 400,
    "max_epoch": 100,
    "batch_size": 8,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "augment_random_verticalflip": 1,
    "augment_random_horizontalflip": 1,
    "augment_random_rotate": 1
})


cfg_xray = Config({
    "train_dir": "",  ## Path to image folder
    "label_dir": "",  ## Path to label folder
    "class_names": {1: "chest"},
    "class_maps": {255: 1},
    "image_size": 400,
    "max_epoch": 100,
    "batch_size": 8,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "augment_random_horizontalflip": 1,
    "augment_random_shiftscale": 1
})


cfg_us = Config({
    "train_dir": "",  ## Path to image folder
    "label_dir": "",  ## Path to label folder
    "class_names": {1: "fetal"},
    "class_maps": {255: 1},
    "image_size": 400,
    "max_epoch": 100,
    "batch_size": 8,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "augment_random_horizontalflip": 1,
    "augment_random_shiftscalerotate": 1
})
