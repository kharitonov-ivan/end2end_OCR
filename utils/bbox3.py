import math
import time

import cv2
import numpy as np
import os
import torch

# import locality_aware_nms as nms_locality
from PIL import ImageDraw, Image, ImageFont

from . import lanms


class Toolbox:

    @staticmethod
    def polygon_area(poly):
        '''
        compute area of a polygon
        :param poly:
        :return:
        '''
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge) / 2.

    @staticmethod
    def restore_rectangle_rbox(origin, geometry):
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                          np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                          d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                          -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                          -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))
        return np.concatenate([new_p_0, new_p_1])

    @staticmethod
    def rotate(box_List, image):
        n = len(box_List)
        c = 0
        angle = 0
        for i in range(n):
            box = box_List[i]
            y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
            y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
            x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
            x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
            for j in range(4):
                if (box[j][1] == y2):
                    k1 = j
            for j in range(4):
                if (box[j][0] == x2 and j != k1):
                    k2 = j
            c = (box[k1][0] - box[k2][0]) * 1.0 / (box[k1][1] - box[k2][1])
            if (c < 0):
                c = -c
            if (c > 1):
                c = 1.0 / c
            angle = math.atan(c) + angle
        angle = angle / n
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        scale = 1
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image_new = cv2.warpAffine(image, M, (w, h))
        return image_new

    @staticmethod
    def resize_image(im, max_side_len=2400):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        im = cv2.resize(im, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    @staticmethod
    def detect(score_map, geo_map, score_map_thresh=0.5, box_thresh=0.1, nms_thres=0.2, timer=None):
        '''1e-5
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        '''
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        text_box_restored = Toolbox.restore_rectangle_rbox(xy_text[:, ::-1] * 4,
                                                           geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start
        # nms part
        start = time.time()
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        timer['nms'] = time.time() - start
        if boxes.shape[0] == 0:
            return None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]
        return boxes, timer

    @staticmethod
    def sort_poly(p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    @staticmethod
    def change_box(box_List):
        n = len(box_List)
        for i in range(n):
            box = box_List[i]
            y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
            y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
            x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
            x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
            box[0][1] = y1
            box[0][0] = x1
            box[1][1] = y1
            box[1][0] = x2
            box[3][1] = y2
            box[3][0] = x1
            box[2][1] = y2
            box[2][0] = x2
            box_List[i] = box
        return box_List

    @staticmethod
    def save_box(box_List, image, img_path):
        n = len(box_List)
        box_final = []
        for i in range(n):
            box = box_List[i]
            y1_0 = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))
            y2_0 = int(max(box[0][1], box[1][1], box[2][1], box[3][1]))
            x1_0 = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))
            x2_0 = int(max(box[0][0], box[1][0], box[2][0], box[3][0]))
            y1 = max(int(y1_0 - 0.1 * (y2_0 - y1_0)), 0)
            y2 = min(int(y2_0 + 0.1 * (y2_0 - y1_0)), image.shape[0] - 1)
            x1 = max(int(x1_0 - 0.25 * (x2_0 - x1_0)), 0)
            x2 = min(int(x2_0 + 0.25 * (x2_0 - x1_0)), image.shape[1] - 1)
            image_new = image[y1:y2, x1:x2]

            gray_2 = image_new[:, :, 0]
            gradX = cv2.Sobel(gray_2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray_2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
            blurred = cv2.blur(gradX, (2, 2))
            (_, thresh) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
            # closed = cv2.erode(thresh, None, iterations = 1)
            # closed = cv2.dilate(closed, None, iterations = 1)
            closed = thresh
            x_plus = []
            x_left = 1
            x_right = closed.shape[1]
            for jj in range(0, closed.shape[1]):
                plus = 0
                for ii in range(0, closed.shape[0]):
                    plus = plus + closed[ii][jj]
                x_plus.append(plus)

            for jj in range(0, int(closed.shape[1] * 0.5 - 1)):
                if (x_plus[jj] > 0.4 * max(x_plus)):
                    x_left = max(jj - 5, 0)
                    break
            for ii in range(closed.shape[1] - 1, int(closed.shape[1] * 0.5 + 1), -1):
                if (x_plus[ii] > 0.4 * max(x_plus)):
                    x_right = min(ii + 5, closed.shape[1] - 1)
                    break

            image_new = image_new[:, x_left:x_right]
            cv2.imwrite("." + img_path.split(".")[1] + '_' + str(i) + ".jpg", image_new)
            box[0][1] = y1
            box[0][0] = x1 + x_left
            box[1][1] = y1
            box[1][0] = x1 + x_right
            box[3][1] = y2
            box[3][0] = x1 + x_left
            box[2][1] = y2
            box[2][0] = x1 + x_right
            box_List[i] = box
        return box_List

    @staticmethod
    def draw_annotation(im, polygons, texts, ttf_font):
        """
        在图片上添加有box和文字
        :param im:  待添加的图片
        :param polygons:     待添加的多边形区域
        :param texts:    待添加的文字
        :param ttf_font:    字体
        :return:    添加完成的图片
        """
        to_return = im.copy()
        draw = ImageDraw.Draw(to_return)
        if polygons is not None:
            for m_polygon, m_text in zip(polygons, texts):
                draw.polygon(xy=m_polygon.flatten().tolist(), outline="green")
                draw.text(m_polygon[2].tolist(), np.unicode(m_text), fill=(255, 0, 255), font=ttf_font)
        else:
            draw.text((15, 90), np.unicode("Not Detected!"), fill=(255, 255, 0), font=ttf_font)
        return to_return

    @staticmethod
    def predict(im_fn, model, with_img, output_dir, with_gpu, labels, output_txt_dir, label_converter,enable_correct=False):
        im = cv2.imread(im_fn.as_posix())[:, :, ::-1]
        im_resized, (ratio_h, ratio_w) = Toolbox.resize_image(im)
        im_resized = im_resized.astype(np.float32)
        im_resized = torch.from_numpy(im_resized)
        if with_gpu:
            im_resized = im_resized.cuda()

        im_resized = im_resized.unsqueeze(0)
        im_resized = im_resized.permute(0, 3, 1, 2)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        score_map, geo_map, (preds, lengths), boxes, pred_mapping, indices = model.forward(im_resized)
        timer['net'] = time.time() - start

        polys = []
        texts = []
        if boxes is not None and len(boxes) > 0:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h
            if preds is not None:
                _, preds = preds.max(2)
                pred_transcripts = []
                for i in range(lengths.numel()):
                    m_text_len = lengths[i]
                    m_text_code = preds[:m_text_len, i]
                    t = label_converter.decode(m_text_code, m_text_len)
                    pred_transcripts.append(t)
                boxes = boxes[indices]
            else:
                pred_transcripts = ['']*len(boxes)

            for m_box, m_pred_transcript in zip(boxes, pred_transcripts):
                m_box = Toolbox.sort_poly(m_box.astype(np.int32))
                if np.linalg.norm(m_box[0] - m_box[1]) < 5 or np.linalg.norm(m_box[3] - m_box[0]) < 5:
                    # print('wrong direction')
                    continue
                poly = np.array([[m_box[0, 0], m_box[0, 1]], [m_box[1, 0], m_box[1, 1]], [m_box[2, 0], m_box[2, 1]],
                                 [m_box[3, 0], m_box[3, 1]]])

                p_area = Toolbox.polygon_area(poly)
                if p_area > 0:
                    poly = poly[(0, 3, 2, 1), :]

                polys.append(poly)
                texts.append(m_pred_transcript)

            if with_img:
                font_file_path = os.path.join(os.path.dirname(__file__),"HanYiXiaoBoHuaYueYuan-Jian-2.ttf")
                ttf_font = ImageFont.truetype(font_file_path, 20)
                im = np.array(Toolbox.draw_annotation(Image.fromarray(im), polys, texts, ttf_font))

        if output_txt_dir is not None:
            gt = output_txt_dir / im_fn.with_name('res_{}'.format(im_fn.stem)).with_suffix('.txt').name

            with gt.open(mode='a', encoding='utf-8') as f:
                if boxes is not None:
                    for m_box in boxes:
                        m_box = np.array(m_box, dtype=np.int32).reshape([1, 8])[0]
                        m_box = [str(x) for x in m_box]
                        bboxstr = ','.join(m_box) + '\n'
                        f.write(bboxstr)
        if labels is not None:
            if boxes is not None:
                res = Toolbox.comp_gt_and_output(boxes, labels, 0.5)
            else:
                res = (0, len(labels), 0)
        else:
            res = (0, 0, 0)
        if output_dir:
            img_path = output_dir / im_fn.name
            cv2.imwrite(img_path.as_posix(), im[:, :, ::-1])
        return polys, im, res

    @staticmethod
    def get_images_for_test(test_data_path):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(test_data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        # print('Find {} images'.format(len(files)))
        return files

    @staticmethod
    def cal_IOU(box1, box2):
        """
        box1, box2: list or numpy array of size 4*2 or 8, h_index first
        """
        box1 = np.array(box1, dtype=np.int32).reshape([1, 4, 2])
        box2 = np.array(box2, dtype=np.int32).reshape([1, 4, 2])

        box1_max = box1.max(axis=1)
        box2_max = box2.max(axis=1)
        w_max = max(box1_max[0][0], box2_max[0][0])
        h_max = max(box1_max[0][1], box2_max[0][1])
        canvas = np.zeros((h_max + 1, w_max + 1))
        # print(canvas.shape)
        box1_canvas = canvas.copy()
        box1_area = np.sum(cv2.drawContours(box1_canvas, box1, -1, 1, thickness=-1))
        # print(box1_area)
        box2_canvas = canvas.copy()
        box2_area = np.sum(cv2.drawContours(box2_canvas, box2, -1, 1, thickness=-1))
        # print(box2_area)
        cv2.drawContours(canvas, box1, -1, 1, thickness=-1)
        cv2.drawContours(canvas, box2, -1, 1, thickness=-1)
        union = np.sum(canvas)
        # print(union)
        intersction = box1_area + box2_area - union
        return intersction / union

    @staticmethod
    def comp_gt_and_output(my_labels, gt_labels, threshold):
        """
        return: [true_pos, false_pos, false_neg]
        """
        coor = gt_labels["coor"]
        ignore = gt_labels["ignore"]
        true_pos, true_neg, false_pos, false_neg = [0] * 4
        for my_label in my_labels:
            for gt_label in coor:
                if Toolbox.cal_IOU(my_label, gt_label) > threshold:
                    true_pos += 1
                    break
            else:
                false_pos += 1
        for i, gt_label in enumerate(coor):
            if ignore[i]:
                continue
            for my_label in my_labels:
                if Toolbox.cal_IOU(gt_label, my_label) > threshold:
                    break
            else:
                false_neg += 1
        return true_pos, false_pos, false_neg


if __name__ == "__main__":
    pass
