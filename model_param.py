
import torch
import torchvision.models as models
import torch.nn as nn
from reshape import reshape_model
#model = models.mobilenet_v3_large(pretrained=True, progress=True)
def model_(arch, num_classes, gpu, batch_size, lr, momentum, weight_decay,
        workers,ngpus_per_node, pretrained=True, dist=False):

    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(arch))
        model = models.__dict__[arch]()

    # reshape the model for the number of classes in the dataset
        model = reshape_model(model, arch, num_classes)

    # transfer the model to the GPU that it should be run on
    if dist:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(batch_size / ngpus_per_node)
            workers = int(workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    return model, optimizer, criterion

