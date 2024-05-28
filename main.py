# -*- coding: utf-8 -*-
import argparse
import torch.nn as nn
from dataset import get_dataloader
from train import train_model
from model import Model
import os
from dateutil import tz
from datetime import datetime
import sys
from dataset_load import load_data
#from model import UnimodalRegressorSequence as Model

class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel', action='store_true',help='whether use DataParallel')
    parser.add_argument('--feature_set', default="baichuan13B-base", type=str)
    parser.add_argument('--fea_dim', default=5120, type=int)
    parser.add_argument('--dataset_file_path', default="/mnt/data/release/")
    parser.add_argument('--classnum', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='size of a mini-batch')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')


    args = parser.parse_args()
    if not os.path.exists("./log"):
        os.makedirs("./log")
    args.log_file_name = '{}_[{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature_set, args.fea_dim,args.lr, args.batch_size)
    sys.stdout = Logger(os.path.join("./log", args.log_file_name + '.log'))

    data = load_data(args)

    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args,data)
    model = Model(args)

    if args.parallel:
        model = nn.DataParallel(model)

    model = model.cuda()
    print('=' * 50)

    train_model(model, train_dataloader,dev_dataloader, test_dataloader, args.epochs, args.lr, args.log_file_name)

    print('=' * 50)

if __name__ == "__main__":
    main()
