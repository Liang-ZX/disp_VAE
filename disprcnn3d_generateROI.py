from typing import Dict, List
import os
import torch
from torch import nn
from torchvision import utils as vutils

from disprcnn.layers import interpolate, ROIAlign
from disprcnn.modeling.pointnet_module.point_rcnn.lib.net.point_rcnn import PointRCNN
from disprcnn.modeling.psmnet.stackhourglass import PSMNet
from disprcnn.modeling.roi_heads.mask_head.inference import Masker
from disprcnn.structures.bounding_box import BoxList
from disprcnn.structures.disparity import DisparityMap
from disprcnn.structures.image_list import ImageList
from disprcnn.utils.stereo_utils import EndPointErrorLoss, expand_box_to_integer


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=input_tensor.device)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=input_tensor.device)
    input_tensor.mul_(std[None, :, None, None]).add_(mean[None, :, None, None])

    file_path = "/home/liangzx/code/disprcnn_plus/models/kitti/roi_result/" + filename +".png"
    vutils.save_image(input_tensor, file_path)

class DispRCNN3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if self.cfg.MODEL.DISPNET_ON:
            self.dispnet = PSMNet(maxdisp=cfg.MODEL.DISPNET.MAX_DISP,
                                  mindisp=cfg.MODEL.DISPNET.MIN_DISP,
                                  is_module=False,
                                  single_modal_weight_average=cfg.MODEL.DISPNET.SINGLE_MODAL_WEIGHTED_AVERAGE)
            self.dispnet_lossfn = EndPointErrorLoss()
            self.disp_resolution = self.cfg.MODEL.DISPNET.RESOLUTIONS[0]
            self.roi_align = ROIAlign((self.disp_resolution, self.disp_resolution), 1.0, 0)
            self.masker = Masker(0.7, 1)
            if self.cfg.MODEL.DISPNET.TRAINED_MODEL != '':
                self.dispnet.load_state_dict(torch.load(
                    self.cfg.MODEL.DISPNET.TRAINED_MODEL, 'cpu'
                )['model'])
                print('Loading PSMNet from', self.cfg.MODEL.DISPNET.TRAINED_MODEL)
        if cfg.MODEL.DET3D_ON:
            self.pcnet = PointRCNN(cfg)
            if self.cfg.MODEL.POINTRCNN.TRAINED_MODEL != '':
                print('loading pointrcnn from', self.cfg.MODEL.POINTRCNN.TRAINED_MODEL)
                ckpt = torch.load(
                    self.cfg.MODEL.POINTRCNN.TRAINED_MODEL, 'cpu'
                )['model']
                sd = {k[7:]: v for k, v in ckpt.items() if k.startswith('module.')}
                self.pcnet.load_state_dict(sd)

    def crop_and_transform_roi_img(self, im, rois_for_image_crop):
        rois_for_image_crop = torch.as_tensor(rois_for_image_crop, dtype=torch.float32, device=im.device)
        im = self.roi_align(im, rois_for_image_crop)
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=im.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=im.device)
        im.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return im

    def generate_roi(self, left_images: ImageList, left_targets: List[BoxList]):
        ims_per_batch = len(left_targets)
        rois_for_image_crop_left = []
        for i in range(ims_per_batch):
            for j, leftbox in enumerate(left_targets[i].bbox.tolist()):
                # 1 align left box and right box
                x1, y1, x2, y2 = expand_box_to_integer(leftbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                y2 = min(y2, left_targets[i].height - 1)
                x2 = min(x2, left_targets[i].width - 1)

                rois_for_image_crop_left.append([i, x1, y1, x2, y2])
        # crop and resize images
        left_roi_images = self.crop_and_transform_roi_img(left_images.tensors, rois_for_image_crop_left)
        if len(left_roi_images) == 0:
            left_roi_images = torch.empty((0, 3, self.disp_resolution, self.disp_resolution)).cuda()
        return left_roi_images

    def _forward_eval(self, left_images: ImageList, right_images: ImageList,
                      left_result: List[BoxList], right_result: List[BoxList], left_targets: List[BoxList], img_ids):
        if self.cfg.MODEL.DISPNET_ON:
            left_roi_images = self.generate_roi(left_images, left_targets)
            dir_path = ("models/kitti/roi_result/left/" + img_ids[0])
            # shutil.rmtree(dir_path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            if len(left_roi_images) > 0:
                for i in range(len(left_roi_images)):
                    save_image_tensor(left_roi_images[i].unsqueeze(0), "left/" + img_ids[0] + "/" + str(i))
                # output = self.dispnet(left_roi_images, right_roi_images)
                output = torch.zeros((0, self.disp_resolution, self.disp_resolution)).cuda()
            else:
                output = torch.zeros((0, self.disp_resolution, self.disp_resolution)).cuda()
        if self.cfg.MODEL.DET3D_ON:
            left_result, right_result, _ = self.pcnet(left_result, right_result, left_targets)
        result = {'left': left_result, 'right': right_result}
        return result

    def remove_illegal_detections(self, left_result: List[BoxList], right_result: List[BoxList]):
        lrs, rrs = [], []
        for lr, rr in zip(left_result, right_result):
            lk = (lr.bbox[:, 2] > lr.bbox[:, 0] + 1) & (lr.bbox[:, 3] > lr.bbox[:, 1] + 1)
            rk = (rr.bbox[:, 2] > rr.bbox[:, 0] + 1) & (rr.bbox[:, 3] > rr.bbox[:, 1] + 1)
            keep = lk & rk
            lrs.append(lr[keep])
            rrs.append(rr[keep])
        return lrs, rrs

    def forward(self, lr_images: Dict[str, ImageList],
                lr_result: Dict[str, List[BoxList]],
                lr_targets: Dict[str, List[BoxList]] = None, img_ids=0):
        left_images, right_images = lr_images['left'], lr_images['right']
        left_result, right_result = lr_result['left'], lr_result['right']
        left_result, right_result = self.remove_illegal_detections(left_result, right_result)
        return self._forward_eval(left_images, right_images, left_result, right_result, lr_targets['left'], img_ids)

    def load_state_dict(self, state_dict, strict=True):
        super(DispRCNN3D, self).load_state_dict(state_dict, strict)
        if self.cfg.MODEL.DISPNET_ON and self.cfg.MODEL.DISPNET.TRAINED_MODEL != '':
            self.dispnet.load_state_dict(torch.load(
                self.cfg.MODEL.DISPNET.TRAINED_MODEL, 'cpu'
            )['model'])
            print('Loading PSMNet from', self.cfg.MODEL.DISPNET.TRAINED_MODEL)
        if self.cfg.MODEL.POINTRCNN.TRAINED_MODEL != '':
            print('loading pointrcnn from', self.cfg.MODEL.POINTRCNN.TRAINED_MODEL)
            ckpt = torch.load(
                self.cfg.MODEL.POINTRCNN.TRAINED_MODEL, 'cpu'
            )['model']
            sd = {k[7:]: v for k, v in ckpt.items() if k.startswith('module.')}
            self.pcnet.load_state_dict(sd)
