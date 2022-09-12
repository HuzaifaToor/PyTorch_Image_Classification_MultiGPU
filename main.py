#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import random
import warnings

import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from train_validate import *
from model_param import *
from train_val_loader import *
from CM import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#
# parse command-line arguments
#
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-dp', '--data', type=str, metavar='DIR', default='/media/iuna/IUNAHuzaifa/IUNA_AI/extra/FinalScripts/',
                    help='path to dataset')
parser.add_argument('--model-dir', type=str, default='', 
				help='path to desired output directory for saving model '
					'checkpoints (default: current directory)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0



#
# initiate worker threads (if using distributed multi-GPU)
#
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #if args.gpu is not None:
    #    warnings.warn('You have chosen a specific GPU. This will completely '
    #                  'disable data parallelism.')

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


#
# worker thread (per-GPU)
#
def main_worker(gpu, ngpus_per_node, args):

    # If folder doesn't exist, then create it.
    if not os.path.isdir(args.data + 'output'):
        os.makedirs(args.data + 'output')
        os.makedirs(args.data + 'output/logs')
        os.makedirs(args.data + 'output/models')

    elif os.path.isdir(args.data + 'output'):
        if not os.path.isdir(args.data + 'output/logs'):
            os.makedirs(args.data + 'output/logs')
        
        if not os.path.isdir(args.data + 'output/models'):
            os.makedirs(args.data + 'output/models')


    writer = SummaryWriter(args.data + 'output/logs/'+args.arch)

    global best_acc1
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

    ################## data loader ##################

    train_loader, train_sampler, num_classes = dataloader_func(args.data,
                                                    args.batch_size,
                                                    args.resolution, 
                                                    args.workers, 
                                                    args.distributed, 
                                                    dir = 'train'
                                                    )

    val_loader = dataloader_func(args.data,
                    args.batch_size,
                    args.resolution, 
                    args.workers, 
                    args.distributed, 
                    dir = 'valid'
                    )


    ############## Model ############################


    model, optimizer, criterion = model_(args.arch, 
                                    num_classes, 
                                    args.gpu, 
                                    args.batch_size, 
                                    args.lr, 
                                    args.momentum, 
                                    args.weight_decay,
                                    args.workers,
                                    ngpus_per_node, 
                                    pretrained=True, 
                                    dist=False
                                    )
    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if args.evaluate:
        validate(val_loader, model, criterion, num_classes, args)
        return

    # train for the specified number of epochs
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # decay the learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc_train, loss_train = train(train_loader, model, criterion, optimizer, epoch, num_classes, args)

        writer.add_scalar(tag='Training_Loss', scalar_value=loss_train, global_step=epoch)
        writer.add_scalar(tag='Training_Accuracy', scalar_value=acc_train, global_step=epoch)

        # evaluate on validation set
        acc_valid, loss_valid = validate(val_loader, model, criterion, num_classes, args)

        writer.add_scalar(tag='val_Loss', scalar_value=loss_valid, global_step=epoch)
        writer.add_scalar(tag='val_Accuracy', scalar_value=acc_valid, global_step=epoch)


        writer.add_figure("Confusion matrix", createConfusionMatrix(train_loader, model, num_classes), epoch)

        # remember best acc@1 and save checkpoint
        acc1 = acc_valid
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'resolution': args.resolution,
                'num_classes': num_classes,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args, best_filename= args.data + 'output/models/' + args.arch + '_model_best.pth.tar', filename=args.data + 'output/models/' + args.arch + '_checkpoint.pth.tar')


if __name__ == '__main__':
    main()
