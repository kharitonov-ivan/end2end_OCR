import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # TRAIN SET
#     OUTPUT_TRAIN_PATH = '/srv/hd3/data/izakharkin/COCO/lmdb/COCO-Text-train.lmdb'
#     TRAIN_IMGS_DIR = '/srv/hd3/data/izakharkin/COCO/train_words/'
#     TRAIN_LABELS_FILEPATH = '/srv/hd3/data/izakharkin/COCO/train_words_gt.txt'
    
#     TRAIN_IMGS_PATHS = []
#     TRAIN_IMGS_LABELS = []
#     with open(TRAIN_LABELS_FILEPATH, 'r') as file:
#         for line in file:
#             try:
#                 print(line)
#                 img_name, text = line.split(',', 1)
#                 TRAIN_IMGS_PATHS.append(TRAIN_IMGS_DIR+img_name+'.jpg')
#                 TRAIN_IMGS_LABELS.append(text)
#             except ValueError:
#                 continue
            
#     createDataset(
#         outputPath=OUTPUT_TRAIN_PATH,
#         imagePathList=TRAIN_IMGS_PATHS,
#         labelList=TRAIN_IMGS_LABELS
#     )
    
    # VAL SET
    OUTPUT_VAL_PATH = '/srv/hd3/data/izakharkin/COCO/lmdb/COCO-Text-val.lmdb'
    VAL_IMGS_DIR = '/srv/hd3/data/izakharkin/COCO/val_words/'
    VAL_LABELS_FILEPATH = '/srv/hd3/data/izakharkin/COCO/val_words_gt.txt'
    
    VAL_IMGS_PATHS = []
    VAL_IMGS_LABELS = []
    with open(VAL_LABELS_FILEPATH, 'r') as file:
        for line in file:
            try:
                print(line)
                img_name, text = line.split(',', 1)
                VAL_IMGS_PATHS.append(VAL_IMGS_DIR+img_name+'.jpg')
                VAL_IMGS_LABELS.append(text)
            except ValueError:
                continue
            
    createDataset(
        outputPath=OUTPUT_VAL_PATH,
        imagePathList=VAL_IMGS_PATHS,
        labelList=VAL_IMGS_LABELS
    )
