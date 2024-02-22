#Partial implementation taken from https://github.com/Beckschen/TransUNet
#Partial implementation taken from https://github.com/raoyongming/GFNet

import argparse
import datetime
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset_train2D import dataset_train, RandomGenerator
from dataset_test3D import dataset3D

def trainer_dataset(args, model, snapshot_path):
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = dataset_train(base_dir=args.root_path, list_dir=args.list_dir, split="train_ADNI_2D",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_valid = dataset3D(base_dir=args.volume_path, list_dir=args.list_dir, split="valid_ADNI_3D")
    data_loader_val = DataLoader(db_valid, batch_size=1, shuffle=False, num_workers=1)
    print("The length of validation set is: {}".format(len(db_valid)))

    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = -1.0
    max_accuracy = 0.0
    max_accuracy_all3D = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        start_time = time.time()
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # print(image_batch.shape, label_batch.shape)
            loss = model(image_batch, label_batch)
            # loss = ce_loss(outputs, label_batch[:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        ##Evaluation:
        model.eval()
        tot_correct = tot_incorrect = total = 0
        for data in data_loader_val:
            x, y = data
            x = x.to(args.device)
            y = y.to(args.device)
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
                    
            if correct >= 7:
                tot_correct = tot_correct + 1
            else:
                tot_incorrect = tot_incorrect + 1
            total = total + 1
        accuracy = tot_correct / total
        print(total, tot_correct, tot_incorrect)
        print("test/accuracy = {}".format(accuracy))
        
        if accuracy > best_performance: 
            best_performance = accuracy
            best_epoch = epoch_num
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        print("Best accuracy = {}".format(best_performance)) 
        print("Best epoch = {}".format(best_epoch))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
