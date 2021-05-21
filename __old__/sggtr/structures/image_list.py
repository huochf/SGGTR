# Modified from Mask R-CNN benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
# ---------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------------------------------------------
from __future__ import division

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, masks, image_sizes, pos=None):
        """
        Arguments:
            tensors (tensor) [b, c, h, w]
            masks (tensor) [b, h, w]
            image_sizes (list[tuple[int, int]])
            pos: (tensor) [b, c, h, w]
        """
        self.tensors = tensors
        self.masks = masks
        self.image_sizes = image_sizes # [h, w]
        self.pos = pos

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.masks.to(*args, **kwargs)
        cast_pos = None
        if self.pos is not None:
            cast_pos = self.pos.to(*args, **kwargs)
        return ImageList(cast_tensor, cast_mask, self.image_sizes,cast_pos)


    def decompose(self):
        return self.tensors, self.masks, self.image_sizes, self.pos


    def __repr__(self):
        return str(self.tensors)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        device = tensors[0].device
        b, _, h, w = batch_shape
        masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, mask in zip(tensors, batched_imgs, masks):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            mask[: img.shape[1], : img.shape[2]] = False

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, masks, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
