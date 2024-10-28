
import torch
import numpy as np
from lib.sml_functions import find_boundaries, expand_boundaries, BoundarySuppressionWithSmoothing
from lib.utils import build_model, build_pebal_model, download_checkpoint
from torchvision import transforms
import skimage
import os
from pathlib import Path
import wget
import torch.nn.functional as F


class Standardized_max_logit:
    def __init__(self, backbone='ResNet101', weight_path=None, class_num=19, dataset='cityscapes'):
        self.backbone = backbone
        self.model = build_model(backbone=backbone)
        self.class_num = class_num

        self.class_mean = np.load(f'stats/{dataset}_mean.npy', allow_pickle=True).item()
        self.class_var = np.load(f'stats/{dataset}_var.npy', allow_pickle=True).item()
        
        if self.class_mean is None or self.class_var is None:
            raise ValueError("Class mean and var could not be loaded!")

    def getscore_from_logit(self, logit):
        confidence_score, prediction = torch.max(logit, axis=1)
        anomaly_score = 1 - confidence_score
        return anomaly_score

    def standardized_max_logit(self, logit):

        anomaly_score, prediction = logit.detach().max(1)
        
        for c in range(self.class_num):
            anomaly_score = torch.where(
                prediction == c,
                (anomaly_score - self.class_mean[c]) / np.sqrt(self.class_var[c]),
                anomaly_score
            )
        return anomaly_score, prediction

    def anomaly_score(self, image, ret_logit=False):
        logit, dec0 = self.model(image)
        
        anomaly_score, prediction = self.standardized_max_logit(logit)
        self.multi_scale = BoundarySuppressionWithSmoothing(
                    boundary_suppression=True,
                    boundary_width=4,
                    boundary_iteration=4,
                    dilated_smoothing=True,
                    kernel_size=7,
                    dilation=6)

        with torch.no_grad():
            anomaly_score = self.multi_scale(anomaly_score, prediction)

        if ret_logit:
            return anomaly_score, logit, dec0
        return anomaly_score, dec0

class Max_logit:
    def __init__(self, backbone = 'ResNet101', weight_path = None, class_num = 19):
        self.backbone = backbone
        self.model = build_model(backbone=backbone)

    def getscore_from_logit(self, logit):
        confidence_score, prediction = torch.max(logit, axis=1)
        anomaly_score = 1 - confidence_score
        return anomaly_score

    def anomaly_score(self, image, ret_logit = False):
        logit, dec0 = self.model(image)
        anomaly_score = self.getscore_from_logit(logit)

        if ret_logit:
            return  anomaly_score, logit, dec0
        return anomaly_score, dec0

class Energy:
    def __init__(self, backbone = 'WideResNet38', weight_path = None, class_num = 19):
        self.backbone = backbone
        self.model = build_model(backbone=backbone, weight_path = weight_path)

    def getscore_from_logit(self, logit):
        anomaly_score = -(1. * torch.logsumexp(logit, dim=1))
        del logit
        return anomaly_score

    def anomaly_score(self, image, ret_logit=False):
        logit, dec0 = self.model(image)
        anomaly_score = self.getscore_from_logit(logit)
        if ret_logit:
            return anomaly_score, logit, dec0
        return anomaly_score, dec0

"""
Reimplementation for PEBAL (ECCV 2021)
"""
class PEBAL:
    def __init__(self, backbone = 'WideResNet38',  weight_path = None, class_num = 19,):
        self.model = build_pebal_model(backbone = backbone,  class_num = class_num+1)
        self.class_num = class_num
        self.gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)

    def getscore_from_logit(self, logit):
        in_logit = logit[:, :self.class_num]
        anomaly_score = -(1. * torch.logsumexp(in_logit, dim=1))
        anomaly_score = self.gaussian_smoothing(anomaly_score)
        return anomaly_score

    def anomaly_score(self, image, ret_logit = False):
        logit = self.model(image)

        anomaly_score = self.getscore_from_logit(logit)
        anomaly_score = self.gaussian_smoothing(anomaly_score)

        in_logit = logit[:, :self.class_num]

        if ret_logit:
            return  anomaly_score, in_logit

        return anomaly_score

