# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
import random

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F

from sggtr.structures.bounding_box import BoxList


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):

    def __init__(self, min_size, max_size=-1):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size, )
        self.min_size = min_size
        self.max_size = max_size


    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_size != -1:
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)


    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        if isinstance(target, BoxList):
            target = target.resize(image.size)
        return image, target


class RandomSizeCrop(object):

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size


    def __call__(self, img, target):
        w = random.randint(self.min_size, max(self.min_size, min(img.width, self.max_size)))
        h = random.randint(self.min_size, max(self.min_size, min(img.height, self.max_size)))
        region = T.RandomCrop.get_params(img, (h, w)) # (top, left, height, width)
        cropped_image = F.crop(img, *region)
        x1, y1, x2, y2 = region[1], region[0], region[1] + region[3], region[0] + region[2]
        target = target.crop((x1, y1, x2, y2))

        return cropped_image, target


class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob


    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):

    def __init__(self, prob=0.5):
        self.prob = prob


    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None,):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, )


    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):

    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):

    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255


    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class RandomAjustImage(object):

    def __init__(self, p=0.5):
        self.p = p


    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.adjust_brightness(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, target


class RandomSelect(object):

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p


    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)



