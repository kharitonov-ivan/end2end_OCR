# end2end_OCR
Course homework in YSDA

FOTS-dev branch based on this repo (https://github.com/novioleo/FOTS). Please follow updates of the one. End2end training does not complete yet. [His readme](https://translate.google.com/translate?depth=1&hl=en&rurl=translate.google.com&sl=zh-CN&sp=nmt4&tl=en&u=https://github.com/novioleo/FOTS)

## Chekpoints

Download [here](https://drive.google.com/drive/folders/1tp0154QhXeByc8bZhJw_Z-bcEWYGyzZg?usp=sharing) and [here](https://yadi.sk/d/Fu2Ct3ilX3zDVg).

1) Trained on ICDAR 2015, bad result 

2) Trained only on SynthText


## How use

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

## Train

Try:
```
python train.py --config="configs/config.json"
```

## Test

If you want to evaluate the detection performance, simply run
```
python eval.py 
```


## Datasets 
- [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=downloads)
- [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8&com=downloads)
