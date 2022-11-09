""" Additional image losses """
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import math
from model_zoo import VGGEncoder
from torch.nn.modules.loss import _Loss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pystrum.pynd.ndutils as nd
from abc import ABC, abstractmethod
from enum import Enum
import os
from collections import defaultdict, OrderedDict


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    code from https://github.com/voxelmorph/voxelmorph

    Licence :
        Apache License Version 2.0, January 2004 - http://www.apache.org/licenses/
    """

    def __init__(self, win=None):
        self.win = win

    def __call__(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [12] * ndims if self.win is None else self.win #9

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1-torch.mean(cc)


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        loss_pl = 0
        ct_pl = 0
        # input = (input + 1) / 2
        # target = (target + 1) / 2
        if len(input.shape) == 5:
            input_ = input[0].permute(1, 0, 2, 3)
            target_ = target[0].permute(1, 0, 2, 3)

            input_features = self.loss_network(input_.repeat(1, 3, 1, 1))
            output_features = self.loss_network(target_.repeat(1, 3, 1, 1))

            for output_feature, input_feature in zip(output_features, input_features):
                loss_pl += F.mse_loss(output_feature, input_feature)
                ct_pl += 1

            input_ = input[0].permute(2, 0, 1, 3)
            target_ = target[0].permute(2, 0, 1, 3)

            input_features = self.loss_network(input_.repeat(1, 3, 1, 1))
            output_features = self.loss_network(target_.repeat(1, 3, 1, 1))

            for output_feature, input_feature in zip(output_features, input_features):
                loss_pl += F.mse_loss(output_feature, input_feature)
                ct_pl += 1

            input = input[0].permute(3, 0, 1, 2)
            target = target[0].permute(3, 0, 1, 2)

        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))

        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
            ct_pl += 1

        return loss_pl / ct_pl


class VGGLoss(torch.nn.Module):
    def __init__(self, device, feature_layer=35):
        super(VGGLoss, self).__init__()
        # Feature extracting using vgg19
        cnn = torchvision.models.vgg19(pretrained=True).to(device)
        self.features = torch.nn.Sequential(*list(cnn.features.children())[:(feature_layer + 1)])
        self.MSE = torch.nn.MSELoss().to(device)

    def normalize(self, tensors, mean, std):
        if not torch.is_tensor(tensors):
            raise TypeError('tensor is not a torch image.')
        for tensor in tensors:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensors

    def forward(self, input, target):
        ct = 1e-8
        mse_loss = 0
        if len(input.shape) == 5:
            # 3D case:
            for axial_slice in range(input.shape[-1]):
                x = input[:, :, :, :, axial_slice]
                y = target.detach()[:, :, :, :, axial_slice]
                if x.shape[1] == 1:
                    x = x.expand(-1, 3, -1, -1)
                    y = y.expand(-1, 3, -1, -1)
                # [-1: 1] image to  [0:1] image---------------------------------------------------(1)
                x = (x+1) * 0.5
                y = (y+1) * 0.5
                # https://pytorch.org/docs/stable/torchvision/models.html
                x.data = self.features(self.normalize(x.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                y.data = self.features(self.normalize(y.data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
                mse_loss += F.mse_loss(x, y.data)
        return mse_loss / ct
