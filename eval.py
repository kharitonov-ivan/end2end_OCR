#!/usr/local/bin/python
# -*-coding=utf-8 -*-
import argparse
import logging
import os
import pathlib

import torch
from tqdm import tqdm

from model.model import FOTSModel
from utils import common_str
from utils.bbox import Toolbox
from utils.util import strLabelConverter

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path)
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    config = checkpoints['config']
    state_dict = checkpoints['state_dict']

    model = FOTSModel(config)
    model.load_state_dict(state_dict)

    if with_gpu:
        model.to(torch.device("cuda:0"))
        model.parallelize()

    model.eval()
    return model


def load_annotation(gt_path):
    with gt_path.open(mode='r') as f:
        label = dict()
        label["coor"] = list()
        label["ignore"] = list()
        label["texts"] = list()  # !!!
        for line in f:
            text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
            if text[8] == "###" or text[8] == "*":
                label["ignore"].append(True)
            else:
                label["ignore"].append(False)
            label["texts"].append(text[8])  # !!!
            bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            label["coor"].append(bbox)
    return label


def main(args: argparse.Namespace):
    model_path = args.model
    image_dir = args.image_dir
    output_img_dir = args.output_img_dir
    output_txt_dir = args.output_txt_dir

    if output_img_dir is not None and not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if output_txt_dir is not None and not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)

    annotation_dir = args.annotation_dir
    with_image = True if output_img_dir else False
    with_gpu = True if torch.cuda.is_available() and not args.no_gpu else False

    true_word_count = 0
    total_words = 0

    model = load_model(model_path, with_gpu=False)
    if annotation_dir is not None:
        print("start evaluation")
        true_pos, true_neg, false_pos, false_neg = [0] * 4

        image_true_word_count, image_word_count, levenshtein = 0, 0, 0

        for image_fn in tqdm(sorted(image_dir.glob('*.jpg'))):
            print('-----------------------------------------------')
            print('\nImage name:', image_fn)
            print()
            gt_path = annotation_dir / image_fn.with_name('gt_{}'.format(image_fn.stem)).with_suffix('.txt').name
            print('Annotation name:', gt_path)
            print()
            labels = load_annotation(gt_path)
            print(labels["texts"])  # !!!
            print()
            # try:
            with torch.no_grad():
                polys, im, res = Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, labels,
                                                 output_txt_dir, strLabelConverter(getattr(common_str, args.keys)))
            true_pos += res[0]
            false_pos += res[1]
            false_neg += res[2]
            image_true_word_count = res[3]

            levenshtein_img = res[4]
            levenshtein += levenshtein_img
            print("res: ", res)

            true_word_count += image_true_word_count
            image_word_count = len([x for x in labels['texts'] if x != '###' and x != '*'])
            total_words += image_word_count

            print('Image true word count:', image_true_word_count)
            print('Image not ignored word count:', image_word_count)
            print(
                'Image word accuracy (for not ignored):',
                image_true_word_count / image_word_count if image_word_count else 'all ignored'
            )

            print(
                'Image word accuracy (only for true positive):',
                image_true_word_count / res[0] if res[0] else 0
            )
            print(
                'Percent of correct symbols:',
                levenshtein_img / res[0] if res[0] else 0
            )

            print()

        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        word_accuracy = true_word_count / total_words

        word_accuracy_2 = true_word_count / true_pos
        levenshtein = levenshtein / true_pos
        print(
            "TP: %d, FP: %d, FN: %d, precision: %f, recall: %f, word accuracy: %f, new word accuracy(only fot tp): %f, percent of correct symbols: %f" % (
                true_pos, false_pos, false_neg, precision, recall, word_accuracy, word_accuracy_2, levenshtein)
            )

    else:
        with torch.no_grad():
            for image_fn in tqdm(image_dir.glob('*.jpg')):
                Toolbox.predict(image_fn, model, with_image, output_img_dir, with_gpu, None, None,
                                strLabelConverter(getattr(common_str, args.keys)))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model',
                        default='./model_best.pth.tar',
                        type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_img_dir', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-t', '--output_txt_dir', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--image_dir', default='/mnt/disk1/dataset/icdar2015/4.4/test/ch4_test_images',
                        type=pathlib.Path,
                        help='dir for input images')
    parser.add_argument('-a', '--annotation_dir',
                        type=pathlib.Path,
                        help='dir for input images')
    parser.add_argument('-k', '--keys',
                        type=str,
                        help='keys in common_str', default="alphabet_and_number")
    parser.add_argument('--no_gpu',
                        action='store_true',
                        help='keys in common_str', default=True)

    args = parser.parse_args()
    main(args)
