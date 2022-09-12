
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

############### Define Data_loader Function ####################

def dataloader_func(path, batch_size, resolution, workers = 2, dist = False, dir = 'train'):


    traindir = os.path.join(path, 'data/train')
    valdir = os.path.join(path, 'data/val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if dir == 'train':

        train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.Resize(224),
            #transforms.RandomResizedCrop(resolution),
            #transforms.RandomHorizontalFlip(15),
            transforms.ToTensor(),
            #normalize,
        ]))

        num_classes = len(train_dataset.classes)
        print('=> dataset classes:  ' + str(num_classes) + ' ' + str(train_dataset.classes))

        if dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        return torch.utils.data.DataLoader(
                                train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                num_workers=workers, pin_memory=True, sampler=train_sampler
                                ), train_sampler, num_classes


    elif dir == 'valid':

        return torch.utils.data.DataLoader(
                    datasets.ImageFolder(valdir, transforms.Compose([
                    #transforms.Resize(256),
                    #transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    #normalize,
                    ])),
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

