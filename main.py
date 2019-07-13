import logging
import argparse
import os
import shutil
import sys
import time
import random
import numpy as np
import pandas as pd
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms


from network import SSR_Net
from datasets import MyDatasets

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.")
    ## training & optimizer
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', dest='epochs', help='Maximum number of training epochs.',
            default=90, type=int)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')        
    parser.add_argument('--batch_size', dest='batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
            default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')   
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    ## distribution
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://192.168.68.58:22', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
	## dataset
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='imdb', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.', default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to filename_list containing lables for every example.', default='', type=str)   
    parser.add_argument('--test_data_dir', dest='test_data_dir', help='Path to test dataset', default='', type=str)
    parser.add_argument('--test_filename_list', dest='test_filename_list', help='Path to test filename_list containing labels for test example.', default='', type=str)
    args = parser.parse_args()
    return args


minmum_loss = np.inf

def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global minmum_loss

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
   
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank) 
    
    # load module
    print('===> creating model SSR_Net')
    model = SSR_Net.SSR_Net(stage_num=[3,3,3],lambda_local=1,lambda_d=1, age=True)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        ## DataParallel use set gups
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    """
	optimizer = torch.optim.SGD(model.parameters(),
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)  
    """  
    # define optimizer 
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    criterion = nn.L1Loss().cuda(args.gpu)

    # load resume from a checkpoint file
    if args.resume == '':
        print("===> training from scrath")
    else:
        if os.path.isfile(args.resume):	
            print("===> loading checkpoint '{}'".format(args.resume))		
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])	
            print("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.resume, checkpoint['epoch']))            						
        else:
            print("===> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ## loading datasets
    print("===> Creat dataloader...")
    transformations = transforms.Compose([
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])])

    if args.dataset == 'imdb':
        train_dataset = MyDatasets.IMDBWIKI(args.data_dir, args.filename_list, transformations, db='imdb')
    elif args.dataset =='wiki':
        train_dataset = MyDatasets.IMDBWIKI(args.data_dir, args.filename_list, transformations, db='wiki')
    else:
        print ('Error: not a valid dataset name')
        sys.exit()

    # data loader
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=train_sampler)

    test_dataset = MyDatasets.IMDBWIKI(args.test_data_dir, 
                                            args.test_filename_list, 
                                            transformations,db='wiki')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size, 
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        loss = test(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best =  loss < minmum_loss
        minmum_loss = min(loss, minmum_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'minmum_loss': minmum_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best,filename = os.path.join('checkpoints','SSR_Net_MT' +  '_' + str(epoch) + '.pth'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    mae =  AverageMeter('MAE', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,mae, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    end = time.time()

    for i, (images, age, gender) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            age = torch.FloatTensor(age)
            images = images.cuda(args.gpu, non_blocking=True)
            age = age.cuda(args.gpu, non_blocking=True)
            gender = gender.cuda(args.gpu, non_blocking=True)

        # compute output
        pred_age = model(images)

        # print("pred_age",pred_age)
        # print("age",age)
        loss_mae = criterion(pred_age, age)
        
        # measure accuracy and record loss
        mae.update(loss_mae.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_mae.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    mae =  AverageMeter('MAE', ':.4e')
    progress = ProgressMeter(len(test_loader), batch_time, mae, prefix="Test:")

    # switch to eval mode  
    model.eval()
    with torch.no_grad():
        end = time.time() 
        for i, (images, age, gender) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                age = age.cuda(args.gpu, non_blocking=True)
                gender = gender.cuda(args.gpu, non_blocking=True)
            
            # compute output
            pred_age = model(images)
            loss_mae = criterion(pred_age, age)
            
            # measure accuracy and record loss
            mae.update(loss_mae.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        return loss_mae.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ =='__main__':
    main()