### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn
from torch.nn import CTCLoss


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        return

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask):
        classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return torch.mean(L_g * y_true_cls * training_mask) + classification_loss

    def __dice_coefficient(self, y_true_cls, y_pred_cls,
                           training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
        union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        if torch.__version__ == "1.1.0":
            self.ctc_loss = CTCLoss(zero_infinity = True)  # pred, pred_len, labels, labels_len
            print("OK")
        else:
            self.ctc_loss = CTCLoss()
            print("ERROR")




    def forward(self, *input):
        gt, pred = input[0], input[1]
        input_seq, target_seq, input_lengths, target_lengths = \
            pred[0], gt[0], pred[1], gt[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_seq = input_seq.cpu()
        # input_lengths =  torch.tensor(input_lengths).int().to(device)
        # target_lengths = torch.tensor(target_lengths).int().to(device)
        # target_seq = torch.tensor(target_seq).int().to(device)

        # ctc_loss(input, target, input_lengths, target_lengths)
        loss = self.ctc_loss(input_seq, target_seq, input_lengths, target_lengths)
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['mode']
        self.detection_loss = DetectionLoss()
        self.recognition_loss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                y_true_recog, y_pred_recog,
                training_mask):

        recognition_loss = torch.tensor([0.0]).float()
        detection_loss = torch.tensor([0.0]).float()

        if self.mode == 'recognition':
            recognition_loss = self.recognition_loss(y_true_recog, y_true_recog, training_mask)
        elif self.mode == 'detection':
            detection_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                 y_true_geo, y_pred_geo, training_mask)
        elif self.mode == 'united':
            detection_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                y_true_geo, y_pred_geo, training_mask)
            if y_true_recog:
                recognition_loss = self.recognition_loss(y_true_recog, y_pred_recog)

        recognition_loss = recognition_loss.to(detection_loss.device)
        return detection_loss, recognition_loss
