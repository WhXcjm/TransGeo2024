import torch
import train
from model.TransGeo import TransGeo
import time
import numpy as np
import argparse
from dataset.CVUSA_ import CVUSA
from dataset.VIGOR import VIGOR

import train
from train import validate
checkpoint = torch.load('C:/Users/xusir/Desktop/CVPR_subset/model_best.pth.tar')
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict=checkpoint['state_dict']

for k, v in state_dict.items():
    name = k#[7:] if k.startswith('module.') else k  # 移除`module.`前缀
    new_state_dict[name] = v

#print(new_state_dict)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# moco specific configs:
parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')

# options for moco v2
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--cross', action='store_true',
                    help='use cross area')

parser.add_argument('--dataset', default='cvusa', type=str,
                    help='vigor, cvusa, cvact')
parser.add_argument('--op', default='adam', type=str,
                    help='sgd, adam, adamw')

parser.add_argument('--share', action='store_true',
                    help='share fc')

parser.add_argument('--mining', action='store_true',
                    help='mining')
parser.add_argument('--asam', action='store_true',
                    help='asam')

parser.add_argument('--rho', default=0.05, type=float,
                    help='rho for sam')
parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')

parser.add_argument('--crop', action='store_true',
                    help='nonuniform crop')

parser.add_argument('--fov', default=0, type=int,
                    help='Fov')








def main():
    print("hhh")
    args = parser.parse_args()
    args.distributed=0
    model = TransGeo(args=args)
    model = torch.nn.DataParallel(model)
    pos_embed_reshape = new_state_dict['module.reference_net.pos_embed'][:, 2:, :].reshape(
        [1,
         np.sqrt(new_state_dict['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
         np.sqrt(new_state_dict['module.reference_net.pos_embed'].shape[1] - 2).astype(int),
         model.module.reference_net.embed_dim]).permute((0, 3, 1, 2))
    args.sat_res=256
    new_state_dict['module.reference_net.pos_embed'] = \
        torch.cat([new_state_dict['module.reference_net.pos_embed'][:, :2, :],
                   torch.nn.functional.interpolate(pos_embed_reshape, (

                       args.sat_res // model.module.reference_net.patch_embed.patch_size[0],
                       args.sat_res // model.module.reference_net.patch_embed.patch_size[1]),
                                                   mode='bilinear').permute((0, 2, 3, 1)).reshape(
                       [1, -1, model.module.reference_net.embed_dim])], dim=1)
    model.load_state_dict(new_state_dict)
    model.eval()
    query_dataset = CVUSA(mode='tt_query', print_bool=True, same_area=(not args.cross), args=args)
    reference_dataset = CVUSA(mode='tt_reference', print_bool=True, same_area=(not args.cross), args=args)
    val_query_loader = torch.utils.data.DataLoader(
        query_dataset,batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)  # 512, 64
    val_reference_loader = torch.utils.data.DataLoader(
        reference_dataset,batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)  # 80, 128

    validate(val_query_loader, val_reference_loader, model, args)

if __name__ == '__main__':
    main()

