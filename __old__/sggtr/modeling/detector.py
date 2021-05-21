
import torch
import torch.nn as nn
from torch import Tensor

from sggtr.structures.image_list import ImageList
from sggtr.structures.bounding_box import BoxList


class Detector(nn.Module):

	def __init__(self, ):
		super(Detector, self).__init__()


	def forward(self, tensor_list: ImageList):
		"""
		Arguments:
			tensor_list: ImageList output by backbone, which consists of:
				- tensor_list.tensors: batched feature maps, of shape[b, c, h, w]
                - tensor_list.masks: batched of feature mask, of shape [b, h, w]
                - tensor_list.image_sizes: list of feature map's size, format by [h, w]
                
        Returns:
        	output: a dict of results, which consists of
        	    - output["pred_box_logits"]: ..
        	    - output[""]: ...
		"""
		raise NotImplementedError
