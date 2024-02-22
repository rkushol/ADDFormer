#Partial implementation taken from https://github.com/Beckschen/TransUNet

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_test3D import dataset3D
from networks.fusion_net import FusionNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='./data/ADNI', help='root dir for testing/validation')
parser.add_argument('--dataset', type=str, default='adni', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_adni', help='list dir')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iterations to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--gf_name', type=str, default='gfnet-b', help='version of gfnet')
parser.add_argument('--gf_path', default='./model/gfnet_checkpoint/gfnet-b.pth', help='checkpoint path')
parser.add_argument('--range_start', default=112, type=int, help='start position of extracted coronal slice')
parser.add_argument('--range_end', default=126, type=int, help='end position of extracted coronal slice')                  

args = parser.parse_args()


def inference(args, model):
    db_test = dataset3D(base_dir=args.volume_path, split="test_ADNI_3D", list_dir=args.list_dir)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    print("The length of test set is: {}".format(len(db_test)))
    logging.info("{} test samples are running now...".format(len(test_loader)))


    model.eval()
    all_preds, all_label = [], []
    loss_fct = torch.nn.CrossEntropyLoss()
    tot_correct = tot_incorrect = total = 0
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pos_num = 13
    neg_num = 12
    '''
    for data in test_loader:
        x, y = data
        x = x.to(args.device)
        y = y.to(args.device)
        label = y.detach().cpu().numpy()
        #print(x.shape)
        #print(y.shape)
        correct = 0
        incorrect = 0
        for slice_number in range(args.range_start, args.range_end):
            #temp_x = x[:, :, :, slice_number, :]   #For MNI
            temp_x = x[:, :, :, :, slice_number]    #For FreeSurfer
            with torch.no_grad():                            
                logits = model(temp_x)[0]
                preds = torch.argmax(logits, dim=-1)
            
            if preds == y:
                correct = correct + 1
                temp_correct_preds = preds
            else:
                incorrect = incorrect + 1
                temp_incorrect_preds = preds
         
                
        if correct >= (args.range_end - args.range_start)/2:
            tot_correct = tot_correct + 1
            #if label == 0:
            #    TP += 1
            #if label == 1:
            #    TN += 1
        else:
            tot_incorrect = tot_incorrect + 1
            #if label == 0:
            #    FP += 1
            #if label == 1:
            #    FN += 1
        #print(correct, incorrect) 
        total = total + 1 
        
    accuracy = tot_correct / total
    print("Total Sample:", total, ", Total Correct:", tot_correct, ", Total Incorrect:", tot_incorrect)

    logging.info("\n")
    logging.info("Test Results")
    logging.info("Test Accuracy: %2.5f" % accuracy)
    print("test/accuracy = {}".format(accuracy))
    '''
    sen = TP / (TP + FN)
    spe = TN / (FP + TN)
    precision = TP / (TP + FP)
    F1score = 2 * (precision * sen) / (precision + sen)
    print(TP, FN)
    print(FP, TN)
    print ("Sensitivity = {}".format(sen))
    print ("Specificity = {}".format(spe))
    print ("Precision = {}".format(precision))
    print ("F1score = {}".format(F1score))
    print("Accuracy = {}".format((TP+TN) / (pos_num+neg_num)))
    '''
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    args.is_pretrain = True
    #args.is_pretrain = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # name the same snapshot defined in train script!
    args.exp = 'ADDFormer_' + dataset_name + '_' + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'ADDFormer')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_slice' + str(args.range_start)
    snapshot_path = snapshot_path + '_to' + str(args.range_end)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    net = FusionNet(args).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)


    inference(args, net)


