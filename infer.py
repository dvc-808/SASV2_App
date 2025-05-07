#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import glob
import torch
import zipfile
import warnings
import argparse
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import *
from SASVNet import *
from DatasetLoader import *
from tuneThreshold import *

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "SASVNet")
## Data loader
parser.add_argument('--max_frames',     type=int,   default=500,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=0,      help='Input length to the network for testing. 0 uses the whole files')
parser.add_argument('--num_eval',       type=int,   default=1,      help='Number of segments of input utterence for testing')
parser.add_argument('--num_spk',        type=int,   default=40,     help='Number of non-overlapped bona-fide speakers within a batch')
parser.add_argument('--num_utt',        type=int,   default=2,      help='Number of utterances per speaker within a batch')
parser.add_argument('--batch_size',     type=int,   default=160,    help='batch_size = num_spk*num_utt + num_spf, num_spf = batch_size - num_spk*num_utt')
parser.add_argument('--max_seg_per_spk',type=int,   default=10000,  help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--num_thread',     type=int,   default=10,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=1,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmax",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam",     help='sgd, adam, adamW, or adamP')
parser.add_argument('--scheduler',      type=str,   default="cosine_annealing_warmup_restarts",     help='Learning rate scheduler')
parser.add_argument('--weight_decay',   type=float, default=1e-7,   help='Weight decay in the optimizer')
parser.add_argument('--lr',             type=float, default=1e-4,   help='Initial learning rate')
parser.add_argument('--lr_t0',          type=int,   default=8,      help='Cosine sched: First cycle step size')
parser.add_argument('--lr_tmul',        type=float, default=1.0,    help='Cosine sched: Cycle steps magnification.')
parser.add_argument('--lr_max',         type=float, default=1e-4,   help='Cosine sched: First cycle max learning rate')
parser.add_argument('--lr_min',         type=float, default=0,      help='Cosine sched: First cycle min learning rate')
parser.add_argument('--lr_wstep',       type=int,   default=0,      help='Cosine sched: Linear warmup step size')
parser.add_argument('--lr_gamma',       type=float, default=0.8,    help='Cosine sched: Decrease rate of max learning rate by cycle')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--num_class',      type=int,   default=41,     help='Number of speakers in the softmax layer, 40 or 20 (speaker-classes) + 1 (spoofing-class)') # 41

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="./exp",     help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list')
parser.add_argument('--audio_format',     type=str,   default="",     help='Audio Format')
parser.add_argument('--eval_list',      type=str,   default="",     help='Enroll mail list')
parser.add_argument('--enroll_female_list',  type=str,   default="",     help='Enroll Female list')
parser.add_argument('--enroll_male_list',    type=str,   default="",     help='Enroll Male list')
parser.add_argument('--train_path',     type=str,   default="",     help='Absolute path to the train set')
parser.add_argument('--eval_path',      type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--infer_path',      type=str,   default="",     help='Absolute path to the infer set')
parser.add_argument('--spk_meta_train', type=str,   default="",     help='')
parser.add_argument('--spk_meta_eval',  type=str,   default="",     help='')
parser.add_argument('--musan_path',     type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="",     help='Absolute path to the test set')

## Model definition
parser.add_argument('--num_mels',       type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,   help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--pooling_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--num_out',        type=int,   default=192,    help='Embedding size in the last FC layer')
parser.add_argument('--eca_c',          type=int,   default=1024,   help='ECAPA-TDNN channel')
parser.add_argument('--eca_s',          type=int,   default=8,      help='ECAPA-TDNN model-scale')


args = parser.parse_args()

def main_worker(args):
    # args.gpu = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Load models
    s = SASVNet(**vars(args))
    s = WrappedModel(s).to(device)
    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()
    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

    sc = trainer.infer(**vars(args))




def main():

    args.model_save_path  = args.save_path+"/model"

    if os.path.exists(args.model_save_path): print("[Folder {} already exists...]".format(args.save_path))

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    main_worker(args)

if __name__ == '__main__':
    main()
