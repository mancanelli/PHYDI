import os
import argparse
import random
from multiprocessing import cpu_count

import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, CyclicLR, MultiStepLR, StepLR)

from getModel import getModel
from training import Trainer
from utils.readFile import readFile
from utils.dataloaders import (CIFAR10_dataloader, CIFAR100_dataloader, STL10_dataloader, SVHN_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1656079)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--n_workers', default=1)
    parser.add_argument('--quat_data', type=bool, default=False)
    parser.add_argument('--n', type=int, default=4, help="n parameter for PHM layer")
    parser.add_argument('--optim', type=str, default="SGD")
    parser.add_argument('--scheduler', type=str, default="cosine")
    parser.add_argument('--l1_reg', type=bool, default=False)
    parser.add_argument('--train_dir', type=str, default='./data/', help="Folder containg training data. It must point to a folder with images in it.")
    
    parser.add_argument('--Dataset', type=str, default='SVHN', help='SVHN, CIFAR10')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet20', help='Models: ...')
    parser.add_argument('--fixup', type=bool, default=False)
    parser.add_argument('--kron_weights', type=int, default=None)
    parser.add_argument('--kron_res', type=bool, default=None)
    parser.add_argument('--rezero', type=bool, default=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', default=(0.5, 0.999))
    parser.add_argument('--print_every', type=int, default=50, help='Print Loss every n iterations')
    parser.add_argument('--get_iter_time', type=bool, default=False)
    parser.add_argument('--get_inf_time', type=bool, default=False)
    parser.add_argument('--EpochCheckpoints', type=bool, default=True, help='Save model every epoch. If set to False the model will be saved only at the end')
    
    parser.add_argument('--TextArgs', type=str, default='TrainingArguments.txt', help='Path to text with training settings')
    parse_list = readFile(parser.parse_args().TextArgs)
    
    opt = parser.parse_args(parse_list)
    
    if opt.n_workers=='max':
        n_workers = cpu_count()
    else:
        n_workers = int(opt.n_workers)


    # Set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    

    if opt.dataset == 'SVHN':
        num_classes = 10
        train_loader, test_loader, eval_loader, data_name = SVHN_dataloader(root=opt.train_dir, quat_data=opt.quat_data, 
                                                                            batch_size=opt.batch_size, img_size=opt.img_size, 
                                                                            num_workers=n_workers)
    elif opt.dataset == 'CIFAR10':
        num_classes = 10
        train_loader, test_loader, eval_loader, data_name = CIFAR10_dataloader(root=opt.train_dir, quat_data=opt.quat_data, 
                                                                               batch_size=opt.batch_size, img_size=opt.img_size, 
                                                                               num_workers=n_workers)
    elif opt.dataset == 'CIFAR100':
        num_classes = 100
        train_loader, test_loader, eval_loader, data_name = CIFAR100_dataloader(root=opt.train_dir, quat_data=opt.quat_data, 
                                                                                batch_size=opt.batch_size, img_size=opt.img_size, 
                                                                                num_workers=n_workers)
    elif opt.dataset == 'STL10':
        num_classes = 10
        train_loader, test_loader, eval_loader, data_name = STL10_dataloader(root=opt.train_dir, quat_data=opt.quat_data, 
                                                                             batch_size=opt.batch_size, img_size=opt.img_size, 
                                                                             num_workers=n_workers)
    else:
        RuntimeError('Wrong dataset or not implemented')
    
    
    net = getModel(str_model=opt.model, quat_data=opt.quat_data, n=opt.n, num_classes=num_classes, 
                   fixup=opt.fixup, kron_weights=opt.kron_weights, kron_res=opt.kron_res, rezero=opt.rezero)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of parameters:', params)
    print()
    

    wandb.init(project="rezero-phcnn", dir="./PH-CNN/")
    config = wandb.config
    wandb.watch(net)
    wandb.config.update(opt)
    #wandb.config.update({"params": params})
    

    checkpoint_folder = './PH-CNN/checkpoints/'
    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    

    # Initialize optimizers
    weight_decay_cifar = 5e-4
    weight_decay_custom = 0.0001
    if opt.optim == "SGD":
        optim_name = "SGD"
        optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    if opt.optim == "Adam":
        optim_name = "Adam"
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, 
                               betas=(float(opt.betas[0]), float(opt.betas[1])))
    

    # Add scheduler
    if opt.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
    elif opt.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=50)
    elif opt.scheduler == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
    elif opt.scheduler == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)


    trainer = Trainer(net, optimizer, scheduler, epochs=opt.epochs, quat_data=opt.quat_data, n=opt.n, 
                      use_cuda=opt.use_cuda, gpu_num=opt.gpu_num, print_every=opt.print_every,
                      checkpoint_folder=checkpoint_folder, saveModelsPerEpoch=opt.EpochCheckpoints,
                      get_iter_time=opt.get_iter_time, get_inf_time=opt.get_inf_time, optim_name=optim_name,
                      lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, l1_reg=opt.l1_reg)

    trainer.train(train_loader, eval_loader)
