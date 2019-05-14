import argparse
import json
import logging
import os
import pathlib
import random
from data_loader import SynthTextDataLoaderFactory
from data_loader import OCRDataLoaderFactory
from data_loader.dataset import ICDAR, MyDataset
from logger import Logger
from model.loss import *
from model.model import *
from model.metric import *
from trainer import Trainer
from utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume, config_from_file):
    train_logger = Logger()

    if config['data_loader']['dataset'] == 'icdar2015':
        # ICDAR 2015
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        ICDARDataset2015 = ICDAR(data_root, year='2015')
        data_loader = OCRDataLoaderFactory(config, ICDARDataset2015)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'icdar2017':
        # ICDAR 2017
        data_root = pathlib.Path(config['data_loader']['data_dir'])
        ICDARDataset2017 = ICDAR(data_root, year='2017')
        data_loader = OCRDataLoaderFactory(config, ICDARDataset2017)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'synth800k':
        data_loader = SynthTextDataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()
    elif config['data_loader']['dataset'] == 'mydataset':
        image_root = pathlib.Path(config['data_loader']['image_dir'])
        annotation_dir = pathlib.Path(config['data_loader']['annotation_dir'])
        data_loader = OCRDataLoaderFactory(config,MyDataset(image_root,annotation_dir))
        train = data_loader.train()
        val = data_loader.val()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = eval(config['arch'])(config_from_file)
    model.summary()

    loss = eval(config['loss'])(config['model'])
    metrics = [eval(metric) for metric in config['metrics']]

    finetune_model = config['finetune']

    trainer = Trainer(model, loss, metrics,
                      finetune=finetune_model,
                      resume=resume,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      train_logger=train_logger,
                      toolbox = Toolbox,
                      keys=getattr(common_str,config['model']['keys']),
                      config_from_file = config_from_file)

    trainer.train()


if __name__ == '__main__':
    
    SEED = 1228
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='./config.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
        config_from_file = json.load(open(args.config))
        config['data_loader'] = config_from_file['data_loader']
        config['trainer']=config_from_file['trainer']
        config['validation'] = config_from_file['validation']
        config['lr_scheduler_type'] = config_from_file['lr_scheduler_type']
        config['lr_scheduler_freq'] = config_from_file['lr_scheduler_freq']
        config['lr_scheduler'] = config_from_file['lr_scheduler']
        config['optimizer_type'] = config_from_file['optimizer_type']
        config['optimizer'] = config_from_file['optimizer']

    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        config_from_file = config
        #assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None


    main(config, args.resume, config_from_file)
