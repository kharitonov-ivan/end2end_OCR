import torch.nn as nn
import torch
import math

from base import BaseModel
from utils.bbox import Toolbox
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from .modules.MORAN.morn import MORN
import pretrainedmodels as pm
import torch.optim as optim
import numpy as np
import utils.common_str as common_str
from collections import OrderedDict



class FOTSModel():

    def __init__(self, config):
        self.mode = config['model']['mode']
        self.config = config
        assert self.mode.lower() in ['recognition', 'detection', 'united'], f'模式[{self.mode}]不支持'
        keys = getattr(common_str, config['model']['keys'])
        backbone_network = pm.__dict__['resnet50'](pretrained='imagenet')  # resnet50 in paper
        backbone_network.eval()
        # backbone as feature extractor
        for param in backbone_network.parameters():
            param.requires_grad = config['need_grad_backbone']

        def backward_hook(self, grad_input, grad_output):
            for g in grad_input:
                g[g != g] = 0  # replace all nan/inf in gradients to zero

        if self.config['rectifier'] == True:
            self.MORN = MORN(nc = 32, targetH=config['model']['crnn']['img_h'], targetW=200)

        if not self.mode == 'detection':
            self.conv_rec = shared_conv.SharedConv(backbone_network, config)
            self.nclass = len(keys) + 1
            self.recognizer = Recognizer(self.nclass, config)
            self.recognizer.register_backward_hook(backward_hook)

        if not self.mode == 'recognition':
            self.conv_det = shared_conv.SharedConv(backbone_network, config)
            self.detector = Detector(config)
            self.detector.register_backward_hook(backward_hook)

        self.roirotate = ROIRotate(config['model']['crnn']['img_h'])
        self.buffers = OrderedDict


        # for param in self.detector.parameters():
        #     try:
        #         param.requires_grad = config['need_grad_detector']
        #     except:
        #         param.requires_grad = True

    def available_models(self):
        if self.mode == 'detection':
            return ['conv_det', 'detector']
        elif self.mode == 'recognition':
            return ['conv_rec', 'recognizer']
        else:
            return ['conv_det', 'detector', 'conv_rec', 'recognizer']

    def parallelize(self):
        for m_model in self.available_models():
            setattr(self, m_model, torch.nn.DataParallel(getattr(self, m_model)))

    def to(self, device):
        for m_model in self.available_models():
            setattr(self, m_model, getattr(self, m_model).to(device))

    def summary(self):
        for m_model in self.available_models():
            getattr(self, m_model).summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [{'params': getattr(self, m_model).parameters()} for m_model in self.available_models()],
            **params
        )
        return optimizer

    def train(self):
        for m_model in self.available_models():
            getattr(self, m_model).train()

    def eval(self):
        for m_model in self.available_models():
            getattr(self, m_model).eval()

    def state_dict(self):
        return {f'{m_ind}': getattr(self, m_model).state_dict()
                for m_ind, m_model in enumerate(self.available_models())}

    def load_state_dict(self, sd, models = None):
        if models == None:
            models = self.available_models()
        for m_ind, m_model in enumerate(models):
            getattr(self, m_model).load_state_dict(sd[f'{m_ind}'])

    @property
    def training(self):
        return all([getattr(self, m_model).training for m_model in self.available_models()])

    def parameters(self):
        for m_module in [getattr(self, m_module) for m_module in self.available_models()]:
            for m_para in m_module.parameters():
                yield m_para

    def forward(self, image, boxes=None, mapping=None, text=None):
        """
        :param image:   图像
        :param boxes:   训练的时候gt的boxes
        :param mapping: 训练的时候boxes与图像的映射
        """
        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        def _compute_boxes(_score_map, _geo_map):
            score = _score_map.permute(0, 2, 3, 1)
            geometry = _geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()
            geometry = geometry.detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            _pred_mapping = []
            _pred_boxes = []
            for i in range(score.shape[0]):
                cur_score = score[i, :, :, 0]
                cur_geometry = geometry[i, :, :, ]
                detected_boxes, _ = Toolbox.detect(score_map=cur_score, geo_map=cur_geometry, timer=timer)
                if detected_boxes is None:
                    continue
                num_detected_boxes = detected_boxes.shape[0]

                if len(detected_boxes) > 0:
                    _pred_mapping.append(np.array([i] * num_detected_boxes))
                    _pred_boxes.append(detected_boxes)
            return np.concatenate(_pred_boxes) if len(_pred_boxes) > 0 else [], \
                   np.concatenate(_pred_mapping) if len(_pred_mapping) > 0 else []

        score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices = \
            None, None, (None, torch.Tensor(0)), boxes, mapping, mapping

        if self.mode == 'detection':
            feature_map_det = self.conv_det.forward(image)
            score_map, geo_map = self.detector(feature_map_det)
            if not self.training:
                pred_boxes, pred_mapping = _compute_boxes(score_map, geo_map)
            else:
                pred_boxes,pred_mapping = boxes,mapping

        elif self.mode == 'recognition':
            pred_boxes, pred_mapping = boxes, mapping
            feature_map_rec = self.conv_rec.forward(image)
            rois, lengths, indices = self.roirotate(feature_map_rec, pred_boxes[:, :8], pred_mapping)
            preds = self.recognizer(rois, lengths).permute(1, 0, 2)
            lengths = torch.tensor(lengths).to(device)
        elif self.mode == 'united':
            feature_map_det = self.conv_det.forward(image)
            score_map, geo_map = self.detector(feature_map_det)
            if self.training:
                pred_boxes, pred_mapping = boxes, mapping
                # pred_boxes, pred_mapping = _compute_boxes(score_map, geo_map)

            else:
                pred_boxes, pred_mapping = _compute_boxes(score_map, geo_map)
            if len(pred_boxes) > 0:
                feature_map_rec = self.conv_rec.forward(image)
                rois, lengths, indices = self.roirotate(feature_map_rec, pred_boxes[:, :8], pred_mapping)

                # Raw rois.shape (batch size, 32, 16, width)
                if self.config['rectifier'] == True:
                    rois = self.MORN(rois, test=False, debug=False)

                preds = self.recognizer(rois, lengths).permute(1, 0, 2)
                lengths = torch.tensor(lengths).to(device)
            else:
                preds = torch.empty(1,image.shape[0],self.nclass, dtype=torch.float)
                lengths = torch.ones(image.shape[0])


        return score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices



class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        crnn_config = config['model']['crnn']
        self.crnn = CRNN(crnn_config['img_h'], 32, nclass, crnn_config['hidden'])

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feature_map):
        final = feature_map
        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  512 * 512 了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(geoMap) * 512

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
