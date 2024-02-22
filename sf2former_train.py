#Partial implementation taken from https://github.com/Beckschen/TransUNet
#Partial implementation taken from https://github.com/raoyongming/GFNet

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.linear_fusion_net import FusionNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer3D import trainer_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ADNI', help='root dir for training')
parser.add_argument('--volume_path', type=str, default='./data/ADNI', help='root dir for testing/validation')
parser.add_argument('--dataset', type=str, default='adni', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_adni', help='list dir')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iterations to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size')
parser.add_argument('--gf_name', type=str, default='gfnet-b', help='version of gfnet')
parser.add_argument('--gf_path', default='./model/gfnet_checkpoint/gfnet-b.pth', help='checkpoint path')
parser.add_argument('--range_start', default=112, type=int, help='start position of extracted coronal slice')
parser.add_argument('--range_end', default=126, type=int, help='end position of extracted coronal slice')
parser.add_argument('--output_dir', type=str, default='./model/Fusion_finetune')


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.is_pretrain = True
    args.exp = 'SF2Former_' + dataset_name + '_' + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'SF2Former')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_slice' + str(args.range_start)
    snapshot_path = snapshot_path + '_to' + str(args.range_end)


    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = FusionNet(args).cuda()

    trainer = {'adni': trainer_dataset}
    trainer[dataset_name](args, net, snapshot_path)

