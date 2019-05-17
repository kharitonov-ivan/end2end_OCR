

### Dataset configuration [data_loader] 

| Field | Action | Reference | Remarks | 
|:--:|:--: |:--:|:--:| 
|dataset|Select a specific type of dataset|icdar2015, mydataset|If your own data format is special, add it yourself in `data_loader\dataset.py` | 
|data_dir| Folder|training_gt, training_images| must contain image and ground truth folder | 
|annotation_dir|Label data folder|training_gt| function is not yet complete | 
|batch_size|batch size|32| too big, memory can't hold | 
|shuffle | Random 
Arrangement | true|Don't worry so much, true is right | |workers|Configure datasetloader build efficiency|0|1. The environment parameter needs to be passed inside docker, otherwise it will give an error<br />2. If the efficiency of get_item is compared Low, it is strongly recommended to open a few more, otherwise the gpu will be idle. | 


#### ICDAR2015 

modify the configuration file as follows: 
```json 
{ 
    "data_loader": { 
        "dataset":"icdar2015",
        "data_dir": "/mnt/disk1/dataset/icdar2015/4.4/training", 
        "batch_size": 16, 
        "shuffle": true, 
        "workers": 0 
    } 
} 
``` 
#### Own data 
modification configuration The file is as follows: 
```json 
{ 
  "data_loader": { 
        "dataset":"mydataset", 
        "image_dir": "/data/OCR/own data/own_dataset/training_images", 
        "annotation_dir": "/data/OCR/ Own data /own_dataset/training_gt", 
        "batch_size": 4, 
        "shuffle": true, 
        "workers":0 
    } 
} 
``` 

#### Validation set configuration [validation] 

```json 
"validation": { 
    "validation_split": 0.15,
    "shuffle": true 
} 
``` 

Because training and testing are in a fixed scale, where `validation_split` indicates the proportion of the test set and `shuffle` is the rearrangement. 

#### Trainer Parameters [trainer] 

```json 
"trainer": { 
    "epochs": 10000, 
    "save_dir": "/path/to/save_model", 
    "save_freq": 1, 
    "verbosity": 2, 
    "monitor": "loss", 
    "monitor_mode": "min" 
} 
``` 

Trainer parameters `epochs` indicates the total number of training rounds, `save_dir` indicates the location where the model is stored, and the final model is located at `save_dir/ Under name`, where `name` is the name of the item in the global variable. `save_freq` indicates that the model is stored once every N epochs. `verbosity` is used to set the logger display level, `monitor` and `monitor_mode` are for generating the optimal model `model_best.pth.

- [val_]recall the recall rate defined in 
`metric` - in the Fscore 

example defined in [val_]hmean `metric`, the smaller the desired `loss`, the better. Of course, you can also set the maximum value of `precious`, that is: `monitor` is `precious`, `monitor_mode` is `max`. 

#### FOTSModel parameter [model] 

```json 
"model": { 
    "mode": "united", 
    "scale": 512, 
    "crnn": { 
        "img_h": 16, 
        "hidden": 1024 
    }, 
    "keys": "number_and_dot" 
} 
``` 

> NOTE 
> 
> This piece is not yet complete and can be customized according to individual needs. 

`mode` has three modes to choose from: `recognition` to only train the recognition model, `detection` to only train the detection model, `united` to detect and not to train together. If you need to test a single module, you can choose whether to detect or identify. The default is to train together. 

The `scale` parameter has not been adapted yet, and will be used later to adjust the size of the recognition frame.

The `img_h` in `crnn` is the height of the FeatureMap of the model passed to CRNN after `ROIRotate`, ** here must be a multiple of 8**. `hidden` is the number of hidden layers in BiLSTM in `crnn`, and the specific parameters are adjusted by themselves. 

`keys` is the character set used for the current recognition. If you need to add or view an existing character set, please go to: [common_str.py](./utils/common_str.py) 
